import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional

from models.models import SpoTMamba
from modules.data_handler import DataHandler
from modules.schedulers import EarlyStopScheduler
from modules.result_manager import ResultManager
from utils.utils import set_all_seeds, ExpLogParser
from utils.constants import ExperimentState, EVAL_METRICS, IN_DIM_DICT, STEPS_PER_DAY_DICT
from utils.metrics import RMSE, MAE, MAPE

class ExperimentHandler():
    
    def __init__(self, config: Dict[str, Any],
                 data: DataHandler,
                 result_manager: Optional[ResultManager]=None) -> None:
        
        setting: Dict[str, Any] = config["setting"]
        self.hyperparameter: Dict[str, Any] = config["hyperparameter"]
        self.data = data
        self.result_manager = result_manager
        
        # Fix the random variables with seed.
        set_all_seeds(setting["seed"])
        
        # CUDA Device.
        self.device = torch.device(config["cuda_id"]) if torch.cuda.is_available() else torch.device("cpu")
        
        # Model to be trained.
        self.model_name = setting["model"]
        self.model = self.select_model()
        self.model.to(self.device)
        
        # Early-stopping scheduler
        self.es_scheduler = EarlyStopScheduler(patience=self.hyperparameter["training"]["patience"],
                                               metrics=EVAL_METRICS)
        # Optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.hyperparameter["training"]["lr"],
                                          weight_decay=self.hyperparameter["training"]["weight_decay"])
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=self.hyperparameter["training"]["milestones"],
                                                                 gamma=self.hyperparameter["training"]["lr_decay_rate"])
        
        # Resume Handling
        self.epoch_st = 1
        self.end = False
        if (self.result_manager is not None) and self.result_manager.resume:
            
            log_parser = ExpLogParser(self.result_manager.log_path)
            
            self.end = log_parser.is_end
            if not self.end:
                self.load_model(best=False)
                self.optimizer.load_state_dict(torch.load(self.result_manager.optimizer_path))
                self.lr_scheduler.load_state_dict(torch.load(self.result_manager.lr_scheduler_path))
                
            self.epoch_st = log_parser.last_epoch + 1
            self.es_scheduler.b_epoch = log_parser.best_epoch
            
            best_results = log_parser.best_results
            if len(best_results) > 0:
                self.es_scheduler.b_results = best_results
        
        # Loss function
        self.loss_fn = nn.HuberLoss()
        
    def select_model(self) -> nn.Module:
        
        data_name = self.data.data_name
        model_name = self.model_name
        hyperparameter = self.hyperparameter["model"]
        
        in_dim = IN_DIM_DICT[data_name]
        out_dim = 1
        steps_per_day = STEPS_PER_DAY_DICT[data_name]
        
        if self.model_name == "SpoTMamba":
            model = SpoTMamba(in_dim,
                              out_dim,
                              emb_dim=hyperparameter.get("emb_dim", 32),
                              ff_dim=hyperparameter.get("ff_dim", 256),
                              graph=self.data.graph,
                              steps_per_day=steps_per_day,
                              seed=self.data.seed,
                              num_walks=hyperparameter.get("num_walks", 2),
                              len_walk=hyperparameter.get("len_walk", 20),
                              num_layers=hyperparameter.get("num_layers", 3),
                              dropout=hyperparameter.get("dropout", 0.1))
            
        assert(not (model is None))
        
        return model
    
    def load_model(self, best: bool=True) -> None:
        """
        The path is determined based on the value of the `best` parameter.
        If `best` is True, the function loads the model from the `best_model_path` of the `result_manager`.
        Otherwise, it loads the model from the `model_path` of the `result_manager`.

        Args:
            best (bool, optional): Whether to load the best model or the latest model. Defaults to True.

        Returns:
            None: This function does not return anything.
        """
        
        model_path = self.result_manager.best_model_path if best else self.result_manager.model_path
        self.model.load_state_dict(torch.load(model_path))
    
    def run(self) -> Dict[str, Any]:
        """
        Runs the experiment by training the model for multiple epochs, performing validations,
        and handling early stopping. Returns the test results after training.
        """
        
        self.result_manager.start_train(self.epoch_st)
        
        if self.end:
            print("Experiment is already finished")
            return None
        
        epochs = self.hyperparameter["training"]["epochs"]+1
        valid_epoch = self.hyperparameter["training"]["valid_epoch"]
        for epoch in range(self.epoch_st, epochs):
            
            epoch_last = epoch-1
            validation = (((epoch_last) % valid_epoch == 0) or \
                          ((epoch_last) == epochs)) and epoch_last > 0
            if validation:
                
                stop = self.validation(epoch_last)
                # Early Stopping
                if stop:        
                    line = f"Early stopping at epoch {epoch_last}"
                    self.result_manager.write_log(line, ExperimentState.TEST)
                    break
            
            self.train_epoch()
    
        self.result_manager.end_train()
        test_results = self.test()
        
        return test_results

    def train_epoch(self) -> None:

        self.result_manager.start_epoch()
        self.model.train()
        torch.cuda.empty_cache()

        loss_l = [] # loss list for each epoch
        
        print("lr", self.optimizer.param_groups[0]['lr'])
        train_loader = self.data.train_loader
        for batch in tqdm(train_loader):
            
            torch.cuda.empty_cache()
            output = self.get_predictions(batch)
            
            loss = self.loss_function(output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_l.append(loss.item())
        
        # End of epoch
        train_loss = np.mean(loss_l)
        max_mem = torch.cuda.max_memory_reserved()
        self.result_manager.end_epoch(self.es_scheduler.b_epoch, self.es_scheduler.patience, train_loss, max_mem)
        self.lr_scheduler.step()

        torch.save(self.model.state_dict(), self.result_manager.model_path)
        torch.save(self.optimizer.state_dict(), self.result_manager.optimizer_path)
        torch.save(self.lr_scheduler.state_dict(), self.result_manager.lr_scheduler_path)
        
    def validation(self, epoch: int) -> bool:
        """
        Perform validation for the current epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            bool: A boolean indicating whether the training should stop.

        This function evaluates the model on the validation data loader and calculates the validation results.
        It then calls the `es_scheduler.step()` method to update the early stopping scheduler.
        If the scheduler indicates that the model should be updated, the best model weights are saved to `self.result_manager.best_model_path`.
        The validation results and the best validation results are written to the result manager.
        Finally, the function returns a boolean indicating whether the training should stop.

        Note: This function assumes that `self.model`, `self.data.val_loader`, and `self.es_scheduler` are properly initialized.
        """
        
        model = self.model
        model.eval()
        val_results = self.check_performance(self.data.val_loader)
        update, stop, b_val_results = self.es_scheduler.step(epoch, val_results)
        
        if update:
            torch.save(model.state_dict(), self.result_manager.best_model_path)
        
        self.result_manager.write_results(val_results, ExperimentState.VALIDATION)
        self.result_manager.write_results(b_val_results, ExperimentState.VALIDATION_BEST)
        
        return stop
    
    def test(self) -> Dict[str, Any]:
        """
        Runs a test on the model using the test data.

        This function restores the model from the epoch specified by `self.es_scheduler.b_epoch`, 
        loads the best model weights, and sets the model to evaluation mode. It then calls 
        `self.check_performance` with the test data loader to obtain the test results. The test results 
        are written to the result manager with the experiment state set to `ExperimentState.END`.

        Returns:
            Dict[str, Any]: A dictionary containing the test results.
        """
        
        self.result_manager.write_log(f"Restore model from epoch {self.es_scheduler.b_epoch}", ExperimentState.TEST)
        self.load_model(best=True)
        self.model.eval()
        test_results = self.check_performance(self.data.test_loader)
        self.result_manager.write_results(test_results, ExperimentState.END)
        
        return test_results

    def get_predictions(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """
        Generate predictions for a given batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the input tensor and the target tensor for a batch of data.

        Returns:
            Dict: A dictionary containing the predicted values and the true values.
                - 'y_pred' (torch.Tensor): The predicted values.
                - 'y_true' (torch.Tensor): The true values.
        """
        
        model = self.model
        output = dict()
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = model(x)

        y_pred = self.data.scaler.inverse_transform(y_pred)
        y = self.data.scaler.inverse_transform(y)
        
        output['y_pred'] = y_pred
        output['y_true'] = y
        
        return output
    
    def loss_function(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        Calculates the loss for a given output dictionary.

        Args:
            output (Dict[str, Any]): A dictionary containing the predicted values and the true values.
                - 'y_pred' (torch.Tensor): The predicted values.
                - 'y_true' (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The calculated loss.
        """
        
        y_pred = output['y_pred']
        y_true = output['y_true']
        loss = self.loss_fn(y_pred, y_true)
        
        return loss

    @torch.no_grad()
    def check_performance(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Calculates the performance metrics for the model using the given DataLoader.

        Args:
            loader (DataLoader): The DataLoader containing the data to evaluate the model.

        Returns:
            Dict[str, Any]: A dictionary containing the performance metrics, including the mean loss, RMSE, MAE, and MAPE.
        """
                
        y_preds = np.empty((0,))
        answers = np.empty((0,))
        losses = np.empty((0,))
            
        for batch in tqdm(loader):
            
            torch.cuda.empty_cache()
            output = self.get_predictions(batch)
            
            y_pred = output["y_pred"].detach().cpu().numpy()
            y_true = output["y_true"].detach().cpu().numpy()
            loss = self.loss_function(output).detach().cpu().numpy()
            
            y_preds = np.append(y_preds, y_pred)
            answers = np.append(answers, y_true)
            losses = np.append(losses, loss)
        
        results = dict()
        results['loss'] = np.mean(losses)
        results['RMSE'] = RMSE(answers, y_preds)
        results['MAE'] = MAE(answers, y_preds)
        results['MAPE'] = MAPE(answers, y_preds)
        
        return results
