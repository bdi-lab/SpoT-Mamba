import os
import re
import random
import datetime

import dgl
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd

from typing import Dict, Optional

from utils.constants import ExperimentState, LOG_LINE_PATTERN, EPOCH_LINES, \
                            PERFORMANCE_LINES, LineType, PERFORMANCE_TOKEN

def set_all_seeds(seed) -> None:
    """
    Set the seed for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 

def create_dir(dir_path) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except OSError:
        print("Error: Failed to create the directory.")

###################### CLASSES & FUNCTIONS FOR HANDLING EXPERIMENT RESULTS ######################

class ExpLogLine():
    
    def __init__(self, line: str) -> None:
        
        line = line.strip()
        timestamp, log_level, exp_state_str = re.findall(LOG_LINE_PATTERN, line)[:3]
        content = re.split(LOG_LINE_PATTERN, line)[-1].strip()
        
        # strip '[' and ']'
        timestamp = timestamp[1:-1]
        log_level = log_level[1:-1]
        exp_state_str = exp_state_str[1:-1]
        
        exp_state = ExperimentState.UNDEFINED
        for state in ExperimentState:
            if exp_state_str == state.name:
                exp_state = state
                break
        
        assert exp_state != ExperimentState.UNDEFINED

        self.timestamp: datetime.datetime = datetime.datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
        self.log_level: str = log_level
        self.exp_state: ExperimentState = exp_state
        self.content: str = content
        
        self.line_type = LineType.PLAIN
        if self.exp_state in EPOCH_LINES:
            self.line_type = LineType.EPOCH
        elif self.exp_state in PERFORMANCE_LINES:
            self.line_type = LineType.PERFORMANCE
    
    @property    
    def content_dict(self) -> Optional[Dict[str, object]]:
        
        content = self.content
        content_dict = None
        
        if self.line_type == LineType.EPOCH:
            content_dict = dict([tuple(item.split(": ")) for item in content.split(" / ")])
            
        elif self.line_type == LineType.PERFORMANCE:
            st = content.find(PERFORMANCE_TOKEN)
            content = content[st+len(PERFORMANCE_TOKEN):].strip()
            content_dict = dict([tuple(item.split(": ")) for item in content.split(" / ")])
        
        return content_dict

class ExpLogParser():
    
    def __init__(self, log_path: str) -> None:
        
        assert os.path.exists(log_path)
        
        self.log_path = log_path
        
        with open(log_path, 'r') as f:
            self.lines = f.readlines()
        
        assert len(self.lines) > 0
        
        # Variables below are initialized when they are first accessed (lazy initialization)
        self._start_time: Optional[datetime.datetime] = None
        self._last_time: Optional[datetime.datetime] = None
        self._is_end: Optional[bool] = None
        self._has_no_results: Optional[bool] = None
        self._last_epoch: Optional[int] = None
        self._best_epoch: Optional[int] = None
        self._mean_epoch_time: Optional[float] = None
        self._elapsed_time: Optional[float] = None
        self._runtime: Optional[float] = None
        self._best_results: Optional[Dict[str, float]] = None
        self._test_results: Optional[Dict[str, float]] = None
    
    @property
    def start_time(self) -> datetime.datetime:
        
        if self._start_time is None:
            start_line = ExpLogLine(self.lines[0])
            self._start_time = start_line.timestamp
        
        return self._start_time
    
    @property
    def last_time(self) -> datetime.datetime:
        
        if self._last_time is None:
            last_line = ExpLogLine(self.lines[-1])
            self._last_time = last_line.timestamp
        
        return self._last_time    

    @property
    def is_end(self) -> bool:
        
        if self._is_end is None:
            last_line = ExpLogLine(self.lines[-1])
            self._is_end = last_line.exp_state == ExperimentState.END
        
        return self._is_end

    @property
    def has_no_results(self) -> bool:
        
        if self._has_no_results is None:
            if self._best_epoch is not None:
                self._has_no_results = (self._best_epoch == 0)
            else:
                last_line = ExpLogLine(self.lines[-1])
                self._has_no_results = (last_line.exp_state == ExperimentState.START) or (last_line.exp_state == ExperimentState.CONFIG)
        
        return self._has_no_results
    
    @property
    def last_epoch(self) -> int:
        
        if self._last_epoch is None:
            self.get_last_epoch_line(assign=True)
            
        return self._last_epoch
    
    @property
    def best_epoch(self) -> int:
        
        if self._best_epoch is None:
            self.get_last_epoch_line(assign=True)
            
        return self._best_epoch

    @property
    def mean_epoch_time(self) -> float:
        
        if self._mean_epoch_time is None:
            self.get_last_epoch_line(assign=True)
            
        return self._mean_epoch_time

    @property
    def elapsed_time(self) -> float:
        
        if self._elapsed_time is None:
            self.get_last_epoch_line(assign=True)
            
        return self._elapsed_time

    @property
    def runtime(self) -> float:
        
        if (self._runtime is None) and self.is_end:
            self._runtime = float((self.last_time - self.start_time).total_seconds())
            
        return self._runtime
    
    @property
    def best_results(self) -> Dict[str, float]:
        
        if self._best_results is None:
            self.get_last_val_best_line(assign=True)
            
        return self._best_results

    @property
    def test_results(self) -> Dict[str, float]:
        
        if (self._test_results is None) and self.is_end:

            last_line = ExpLogLine(self.lines[-1])
            self._test_results = dict()
            for metric, value in last_line.content_dict.items():
                self._test_results[metric] = float(value)
            
        return self._test_results
    
    def get_last_epoch_line(self, assign=False) -> Optional[ExpLogLine]:
        
        line = None
        for l in reversed(self.lines):
            
            l = ExpLogLine(l)
            if l.line_type == LineType.EPOCH:
                line = l
                break
        
        if assign:
            self._last_epoch = 0 if line is None else int(line.content_dict["Epoch"][:4])
            self._best_epoch = 0 if line is None else int(line.content_dict["Best Epoch"][:4])
            self._mean_epoch_time = 0 if line is None else float(line.content_dict["Mean Epoch Time"][:-1])
            self._elapsed_time = 0 if line is None else float(line.content_dict["Elapsed Time"][:-1])
        
        return line
    
    def get_last_val_best_line(self, assign=False) -> Optional[ExpLogLine]:
        
        line = None
        for l in reversed(self.lines):
            
            l = ExpLogLine(l)
            if l.exp_state == ExperimentState.VALIDATION_BEST:
                line = l
                break
        
        if assign:
            self._best_results = dict()
            if line is not None:
                
                for metric, value in line.content_dict.items():
                    self._best_results[metric] = float(value)
        
        return line

#################################################################################################


########################################## NumPy Utils ##########################################

def stacked_arange(start: int, end: int, window_size: int):
    '''
    Return a 2D array with shape (num_windows, window_size) by stacking arange arrays
    
    For example:
        >>> stacked_arange(0, 5, 3)
        array([[0, 1, 2],
               [1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])
    '''
    result = np.lib.stride_tricks.sliding_window_view(
        np.arange(start, end+window_size-1), window_size)
    return result

#################################################################################################


###################################### SciPy Sparse Utils #######################################

def csv_to_scipy_sparse(filepath: str, num_nodes: int, format="csr", symmetric=True):
    '''
    Read a csv file and convert it to a scipy sparse matrix
    '''
    df = pd.read_csv(filepath)
    
    row = df['from'].values
    col = df['to'].values
    weight = df['cost'].values
    
    if symmetric:
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        weight = np.concatenate([weight, weight])
    
    adj = sp.coo_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
    
    if format == "csc":
        adj = adj.tocsc()
    elif format == "csr":
        adj = adj.tocsr()
    elif format == "coo":
        pass
    else:
        raise ValueError("format must be csc, csr or coo")
    
    return adj

def to_symmetric(adj: sp.spmatrix):
    '''
    Convert the given adjacency matrix to symmetric matrix
    '''
    adj = adj.copy()
    mask = adj != 0
    adj[mask.T] = adj[mask]
    return adj

def is_symmetric(adj: sp.spmatrix):
    '''
    Check whether the given adjacency matrix is symmetric
    '''
    return (adj - adj.T).sum() == 0

#################################################################################################


######################################### PyTorch Utils #########################################

def stable_softmax(logits, dim):
    """
    Implements a numerically stable version of the softmax function.
    It subtracts the max value in the logits over the specified dimension
    to prevent overflow in the exponential calculation, then computes the softmax.
    
    Parameters:
    - logits (Tensor): The input tensor for which the softmax function will be applied.
    - dim (int): The dimension along which softmax will be computed.
    
    Returns:
    - Tensor: The result of applying the softmax function over the specified dimension.
    """
    
    # Subtract the max value in logits along the specified dimension to prevent overflow
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    stable_logits = logits - max_logits
    
    # Compute softmax over the specified dimension
    return F.softmax(stable_logits, dim=dim)

#################################################################################################
