import os
import time
import logging
import logging.handlers

from typing import Dict

from utils.utils import create_dir, ExpLogParser
from utils.constants import ExperimentState, RESULT_ROOT_DIR

class ResultManager:
    """
    ResultManager manages and saves results of model training and testing based on config.
        - Experiment log txt file is saved in self.log_path.
        - Model checkpoint pickle file is saved in self.model_path.
        - Best model checkpoint pickle file is saved in self.best_model_path.
    """
    
    def __init__(self, config: Dict, root_dir=RESULT_ROOT_DIR) -> None:
        
        setting = config["setting"]
        exp_name = setting["exp_name"]
        dataset = setting["dataset"]
        model = setting["model"]
        in_steps = setting["in_steps"]
        out_steps = setting["out_steps"]
        train_ratio = setting["train_ratio"]
        seed = setting["seed"]
        
        dir_name = f"{dataset}-{in_steps}-{out_steps}-{str(train_ratio).zfill(2)}-{seed}-{model}"
        config_name = config["config_path"].split('/')[-1][:-5]
        
        save_dir = os.path.join(root_dir, exp_name, dir_name)
        log_dir = os.path.join(save_dir, "logs")
        model_dir = os.path.join(save_dir, "saved_models")
        
        create_dir(save_dir)
        create_dir(log_dir)
        create_dir(model_dir)
        
        self.log_path = os.path.join(log_dir, f"{config_name}.log")
        self.model_path = os.path.join(model_dir, f"{config_name}.pickle")
        self.best_model_path = os.path.join(model_dir, f"{config_name}_best.pickle")
        self.optimizer_path = os.path.join(model_dir, f"{config_name}_opt.pickle")
        self.lr_scheduler_path = os.path.join(model_dir, f"{config_name}_lr.pickle")
        
        self.resume = (not config['force_retrain']) and  \
            os.path.exists(self.log_path) and \
            os.path.exists(self.model_path) and \
            os.path.exists(self.optimizer_path)
            
        self.init_log(config)
    
    def init_log(self, config: Dict) -> None:
        
        fmt = '[%(asctime)s] [%(levelname)s]: %(message)s'
        datefmt = '%Y/%m/%d %H:%M:%S'
        filemode = 'a' if self.resume else 'w'
        logging.basicConfig(filename=self.log_path, filemode=filemode,
                            level=logging.INFO, format=fmt, datefmt=datefmt)
        
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        streamHandler = logging.StreamHandler() # For printing to console
        streamHandler.setFormatter(formatter)

        logger = logging.getLogger() # root logger
        logger.addHandler(streamHandler)
        
        if not self.resume:
            for key in config.keys(): 
                self.write_log(f"{key}: {config[key]}", ExperimentState.CONFIG)

    def write_log(self, line: str, exp_state: ExperimentState, log_level=logging.INFO) -> None:
        
        line = f"[{exp_state.name}] {line}"
        logging.log(log_level, line)
    
    def write_results(self, results: Dict[str, float], exp_state: ExperimentState) -> None:
        
        line = "<PERFORMANCE> " + " / ".join([f"{metric}: {value:.2f}" for metric, value in results.items()])
        self.write_log(line, exp_state)

    def start_train(self, epoch_st: int) -> None:
        
        self.start_time = time.time()
        self.elapsed_time = 0
        self.mean_epoch_time = 0
        
        if self.resume:
            log_parser = ExpLogParser(self.log_path)
            self.elapsed_time = log_parser.elapsed_time
            self.mean_epoch_time = log_parser.mean_epoch_time
            self.start_time -= self.elapsed_time
        
        self.epoch = epoch_st
    
    def start_epoch(self) -> None:
        
        self.epoch_start_time = time.time()
    
    def end_epoch(self, b_epoch, patience, loss, max_mem) -> None:
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.elapsed_time = epoch_end_time - self.start_time
        
        # Cumulative moving average of epoch time
        self.mean_epoch_time = float(self.mean_epoch_time*(self.epoch-1) + epoch_time) / float(self.epoch)
        
        line = f"Epoch: {str(self.epoch).zfill(4)} ({self.epoch - b_epoch}/{patience})"
        line += f" / Best Epoch: {str(b_epoch).zfill(4)} / Loss: {loss:.5f}"
        line += f" / Epoch Time: {epoch_time:.3f}s / Mean Epoch Time: {self.mean_epoch_time:.3f}s / Elapsed Time: {self.elapsed_time:.3f}s"
        line += f" / Max GPU Mem: {max_mem/(2.**30):.4f} GiB"
        self.write_log(line, ExperimentState.TRAIN if b_epoch != 0 else ExperimentState.START)
        
        self.epoch += 1
    
    def end_train(self) -> None:
        
        self.elapsed_time = time.time() - self.start_time
        line = f"Total training time: {self.elapsed_time:.4f}s"
        self.write_log(line, ExperimentState.TEST)
    