import sys

from typing import List, Dict, Tuple

from utils.constants import EVAL_METRICS

class EarlyStopScheduler():
    
    def __init__(self, patience, metrics: List[str]=EVAL_METRICS) -> None:
        
        self.patience = patience
        self.metrics = metrics # LB (Lower is Better) metrics only
        
        self.b_epoch = 0
        self.b_results = dict()
        
        for metric in metrics:
            self.b_results[metric] = sys.maxsize
    
    # It is called per validation
    def step(self, epoch: int, results: Dict[str, float]) -> Tuple[bool, bool, Dict[str, float]]:
        
        gain = 0
        
        # results should contain all self.metrics
        for metric in self.metrics:
            
            curr = results[metric]
            best = self.b_results[metric]
            gain += (curr - best) / best
        
        # Renew best results
        update = False
        if gain < 0:
            
            self.b_epoch = epoch
            self.b_results = results
            update = True
        
        stop = (epoch - self.b_epoch) >= self.patience
        
        return update, stop, self.b_results