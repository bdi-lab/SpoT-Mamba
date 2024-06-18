import dgl
import os
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple

from modules.scalers import StandardScaler
from utils.utils import set_all_seeds, csv_to_scipy_sparse, is_symmetric, stacked_arange
from utils.constants import DATA_ROOT_DIR, DIR_NAME_DICT

class DataHandler():
    
    def __init__(self, config: Dict[str, Any]) -> None:
        
        # Arguments for DataHandler.
        setting = config["setting"]
        
        self.data_name: str = setting["dataset"]
        self.model_name: str = setting["model"]
        self.in_steps: int = setting["in_steps"]
        self.out_steps: int = setting["out_steps"]
        self.train_ratio: float = setting["train_ratio"]
        self.val_ratio: float = setting["val_ratio"]
        self.seed: int = setting["seed"]
        
        self.batch_size: int = config["hyperparameter"]["training"]["batch_size"]
        self.device = torch.device(config["cuda_id"]) if torch.cuda.is_available() else torch.device("cpu")
        
        # Fix the random variables with seed.
        set_all_seeds(self.seed)
        
        # Load and preprocess dataset
        data = self.load_data()
        
        self.scaler = StandardScaler()
        sequence = data["sequence"]
        sequence[:, :, 0] = self.scaler.fit_transform(sequence[:, :, 0])
        
        self.sequence: torch.Tensor = torch.tensor(sequence, dtype=torch.float32)
        self.graph: dgl.DGLGraph = data["graph"]
        
        # Generate train/validation/test batches.
        (x_train, y_train,
         x_val, y_val,
         x_test, y_test) = self.generate_batches()

        # Generate DataLoaders.
        self.train_loader = self.generate_loader(x_train, y_train, shuffle=True)
        self.val_loader = self.generate_loader(x_val, y_val, shuffle=False)
        self.test_loader = self.generate_loader(x_test, y_test, shuffle=False)
        
        # Print some statistics.
        print(f"Finished loading dataset and generating DataLoader with {self.data_name}.")
        
        print("="*70)
        print(f"# Nodes: {self.graph.num_nodes():,d} / # Edges: {(self.graph.num_edges()) // 2:,d}")
        print(f"Train X: {str(list(x_train.shape)):>25} / Train Y: {str(list(y_train.shape)):>25}")
        print(f"Validation X: {str(list(x_val.shape)):>19} / Validation Y: {str(list(y_val.shape)):>19}")
        print(f"Test X: {str(list(x_test.shape)):>25} / Test Y: {str(list(y_test.shape)):>25}")
        print("="*70)
    
    def load_data(self, root_dir=DATA_ROOT_DIR) -> Dict[str, Any]:
        """
        Load data for the specified data name.

        Args:
            root_dir (str, optional): The root directory where the data is located. Defaults to DATA_ROOT_DIR.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded data. The dictionary has two keys:
                - "sequence" (np.ndarray): The sequence data loaded from a file with the name "{dir_name}.npz".
                - "graph" (dgl.DGLGraph): The DGL graph loaded from a file with the name "{dir_name}.bin".
                  If the file does not exist, the graph is generated from a CSV file with the name "{dir_name}.csv".
                  The graph is then saved to the file with the name "{dir_name}.bin".
        """

        assert (self.data_name in DIR_NAME_DICT.keys())
        
        dir_name = DIR_NAME_DICT[self.data_name]
        dir_path = os.path.join(root_dir, dir_name)

        sequence: np.ndarray = np.load(os.path.join(dir_path, f"{dir_name}.npz"))["data"]

        graph_path = os.path.join(dir_path, f"{dir_name}.bin")
        
        # Load DGL Graph
        if os.path.exists(graph_path):
            print(f"Load graph from {graph_path}.")
            
            graphs, _ = dgl.load_graphs(graph_path)
            graph: dgl.DGLGraph = graphs[0]
        
        # Generate DGL Graph
        else:
            print(f"Not found graph in {graph_path}. Generate graph data file.")
            
            filepath = os.path.join(dir_path, f"{dir_name}.csv")
            num_nodes = sequence.shape[1]
            adj = csv_to_scipy_sparse(filepath, num_nodes, format="csr", symmetric=True)
            assert is_symmetric(adj)
            
            graph = dgl.from_scipy(adj, eweight_name='cost')
            
            # Min-Max normalization
            cost = graph.edata['cost']
            cost -= cost.min(0, keepdim=True)[0]
            cost /= cost.max(0, keepdim=True)[0]
            graph.edata['cost'] = cost
            
            dgl.save_graphs(graph_path, [graph])
            print(f"Saved graph to {graph_path}.")
        
        graph = dgl.remove_self_loop(graph)
        
        data = dict({"sequence": sequence, "graph": graph})
        
        return data
    
    def generate_batches(self) -> Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
        """
        Generates batches of input and output tensors for training, validation, and testing.
        """
        
        in_steps = self.in_steps
        out_steps = self.out_steps
        
        num_data = self.sequence.shape[0] - (in_steps + out_steps)
        train_end = int(num_data * self.train_ratio/100 + 0.5)
        val_end = train_end + int(num_data * self.val_ratio/100 + 0.5)
        test_end = num_data+1
        
        x_train, y_train = self.generate_batch(0, train_end)
        x_val, y_val = self.generate_batch(train_end, val_end)
        x_test, y_test = self.generate_batch(val_end, test_end)

        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def generate_batch(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of input and output tensors based on the given start and end indices.

        Parameters:
            start (int): The starting index of the batch.
            end (int): The ending index of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor (x) and the output tensor (y).
                - x (torch.Tensor): The input tensor of shape (batch_size, in_steps, num_features).
                - y (torch.Tensor): The output tensor of shape (batch_size, out_steps, num_features).

        """
        
        in_steps = self.in_steps
        out_steps = self.out_steps
        
        x = self.sequence[torch.tensor(stacked_arange(start, end, in_steps), dtype=int)]
        y = self.sequence[:, :, 0:1][torch.tensor(stacked_arange(start+in_steps, end+in_steps, out_steps), dtype=int)]
        
        return x, y
    
    def generate_loader(self, x: torch.Tensor, y: torch.Tensor,
                        shuffle: bool, num_workers: int=0) -> DataLoader:
        """
        Generates a data loader for the given input tensors `x` and `y`.

        Args:
            x (torch.Tensor): The input tensor of shape (num_samples, in_steps, num_features).
            y (torch.Tensor): The output tensor of shape (num_samples, out_steps, num_features).
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.

        Returns:
            DataLoader: A data loader object that provides batches of input and output tensors for training, validation, and testing.
        """
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)
        
        return loader