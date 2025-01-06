# SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces
This code is the official implementation of the following [paper](https://arxiv.org/abs/2406.11244):

> Jinhyeok Choi, Heehyeon Kim, Minhyeong An, and Joyce Jiyoung Whang, SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces, Spatio-Temporal Reasoning and Learning (STRL) Workshop at the 33rd International Joint Conference on Artificial Intelligence (IJCAI 2024), 2024

All codes are written by Jinhyeok Choi (cjh0507@kaist.ac.kr). When you use this code, please cite our paper.

```bibtex
@article{spotmamba,
  author={Jinhyeok Choi and Heehyeon Kim and Minhyeong An and Joyce Jiyoung Whang},
  title={{S}po{T}-{M}amba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces},
  year={2024},
  journal={arXiv preprint arXiv.2406.11244},
  doi={10.48550/arXiv.2406.11244}
}
```

## Requirments
We used Python 3.8, Pytorch 1.13.1, and DGL 1.1.2 with cudatoolkit 11.7.

We also used the official implementation of Mamba (mamba-ssm 1.2.0.post1).

For installation instructions of Mamba, please refer to [the official repository](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation).

## Usage
### SpoT-Mamba
We used NVIDIA GeForce RTX 3090 24GB for all our experiments. We provide the template configuration file (`template.json`).

To train SpoT-Mamba, use the `run.py` file as follows:

```python
python run.py --config_path=./template.json
```
Results will be printed in the terminal and saved in the directory according to the configuration file.

You can find log files and checkpoints resulting from experiments in the `f"experimental_results/{dataset}-{in_steps}-{out_steps}-{str(train_ratio).zfill(2)}-{seed}-{model}"` directory.

### Training from Scratch
To train SpoT-Mamba from scratch, run `run.py` with the configuration file. Please refer to `modules/experiment_handler.py`, `modules/data_handler.py`, and `models/models.py` for examples of the arguments in the configuration file.

The list of arguments of the configuration file:

```json
{
    "setting": {
        "exp_name": "Name of the experiment.",
        "dataset": "The dataset to be used, e.g., 'pems04'.",
        "model": "The model type to be used, e.g., 'SpoTMamba'.",
        "in_steps": "Number of input time steps to use in the model.",
        "out_steps": "Number of output time steps (predictions) the model should generate.",
        "train_ratio": "Percentage of data to be used for training (expressed as an integer out of 100).",
        "val_ratio": "Percentage of data to be used for validation (expressed as an integer out of 100).",
        "seed": "Random seed for the reproducibility of results."
    },
    "hyperparameter": {
        "model": {
            "emb_dim": "Dimension of each embedding.",
            "ff_dim": "Dimension of the feedforward network within the model.",
            "num_walks": "Number of random walks to perform (M).",
            "len_walk": "Length of each random walk (K).",
            "num_layers": "Number of Mamba blocks / Number of layers in the Transformer encoder.",
            "dropout": "Dropout rate used in the model."
        },
        "training": {
            "lr_decay_rate": "Decay rate for learning rate.",
            "milestones": [
                "Epochs after which the learning rate will decay."
            ],
            "epochs": "Total number of training epochs.",
            "valid_epoch": "Number of epochs between each validation.",
            "patience": "Number of epochs to wait before early stopping if no progress on the validation set.",
            "batch_size": "Size of the batches used during training.",
            "lr": "Initial learning rate for training.",
            "weight_decay": "Weight decay rate used for regularization during training."
        }
    },
    "cuda_id": "CUDA device ID (GPU ID) to be used for training if available.",
    "force_retrain": "Flag to force the retraining of the model even if a trained model exists."
}
```

## Hyperparameters
We tuned SpoT-Mamba with the following tuning ranges:

- `emb_dim`: 32
- `ff_dim`: 256
- `num_walks`: {2, 4}
- `len_walk`: 20
- `num_layers`: 3
- `dropout`: 0.1
- `lr_decay_rate`: {0.1, 0.5}
- `milestones`: fixed as [20, 40, 60]
- `epochs`: 300
- `valid_epoch`: 1
- `patience`: 20
- `batch_size`: 32
- `lr`: {0.001, 0.0005}
- `weight_decay`: {0.001, 0.0001}

## Description for each file

### `./`
- `run.py`: The main script to start the model training and evaluation.
- `template.json`: Template for the experiment configuration.

### `./datasets`
- `PEMS04`
  - `PEMS04.bin`: DGLGraph containing pre-processed PEMS04.
  - `PEMS04.csv`: csv file containing the graph structure of PEMS04.
  - `PEMS04.npz`: npz file containing PEMS04.

### `./models`
- `layers.py`: Contains definitions of the neural network layers used in SpoT-Mamba.
- `models.py`: Contains the definition of SpoT-Mamba.

### `./modules`
- `data_handler.py`: Manages data loading and preprocessing.
- `experiment_handler.py`: Handles the setup and execution of experiments.
- `result_manager.py`: Manages the logging and saving of experiment results.
- `scalers.py`: Contains scaler functions for data normalization.
- `schedulers.py`: Contains schedulers for early stopping.

### `trained_models`
- `pems04-60-11-SpoTMamba_best.json`: JSON file containing the best model configuration.
- `pems04-60-11-SpoTMamba_best.log`: Experiment log file of the best model.
- `pems04-60-11-SpoTMamba_best.pickle`: Pickle file containing the checkpoint of the best trained model.

### `utils`
- `constants.py`: Defines constants used across the project.
- `metrics.py`: Contains metrics for evaluating model performance.
- `utils.py`: Contains helper functions used throughout the project.

