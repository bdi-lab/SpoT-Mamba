import torch
import os, argparse, json

from modules.result_manager import ResultManager
from modules.data_handler import DataHandler
from modules.experiment_handler import ExperimentHandler
from utils.utils import ExpLogParser

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/template.json')
    args = vars(parser.parse_args())
    
    return args

def main(args) -> None:
    
    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    torch.cuda.empty_cache()
    
    config_path = args['config_path']
    with open(config_path) as f:
        config = json.load(f)
    
    config['config_path'] = config_path
    
    result = ResultManager(config=config)
    if result.resume:
            
        log_parser = ExpLogParser(result.log_path)
        
        if log_parser.is_end:
            print("Experiment is already finished")
            return
    
    data = DataHandler(config)
    exp = ExperimentHandler(config, data, result)
    exp.run()

if __name__ == '__main__':
    
    args = get_arguments()
    main(args)