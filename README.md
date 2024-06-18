# SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces
This code is the official implementation of the following [paper](https://arxiv.org/abs/2406.11244):

> Jinhyeok Choi, Heehyeon Kim, Minhyeong An, and Joyce Jiyoung Whang, SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces, Spatio-Temporal Reasoning and Learning (STRL) Workshop at the 33rd International Joint Conference on Artificial Intelligence (IJCAI 2024), 2024

All codes are written by Jinhyeok Choi (cjh0507@kaist.ac.kr). When you use this code, please cite our paper.

```bibtex
@article{spotmamba,
  author={Jinhyeok Choi, Heehyeon Kim, Minhyeong An, and Joyce Jiyoung Whang},
  title={SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces},
  year={2024},
  journal={arXiv preprint arXiv.2406.11244},
  doi = {10.48550/arXiv.2406.11244}
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

Each run corresponds to an experiment ID `f"{dataset_name}-{train_ratio}-{seed}-{time}"`.

You can find log files and pandas DataFrame pickle files associated with experiment IDs in the designated directory.

There are some useful functions to handle experiment results in `utils.py`.

You can find an example in `performance_check.ipynb`.

### Training from Scratch
To train SpoT-Mamba from scratch, run `run.py` with the configuration file. Please refer to `modules/experiment_handler.py`, `modules/data_handler.py`, and `models/models.py` for examples of the arguments in the configuration file.

The list of arguments of the configuration file:
- `--seed`: seed

## Hyperparameters
We tuned SpoT-Mamba with the following tuning ranges:
- `lr`: {0.01, 0.001}
- `weight_decay`: {0.001, 0.0001}

## Description for each file
- `datasets.py`: A file for loading the YelpChi and Amazon_new datasets
- `data_handler.py`: A file for processing the given dataset according to the arguments
- `layers.py`: A file for defining the SpoT-MambaConv layer
- `model_handler.py`: A file for training SpoT-Mamba
- `models.py`: A file for defining SpoT-Mamba architecture
- `performance_check.ipynb`: A file for checking the fraud detection performance of SpoT-Mamba
- `run.py`: A file for running SpoT-Mamba on the YelpChi and Amazon_new datasets
- `result_manager.py`: A file for managing train, validation, and test logs
- `template.json`: A template file consisting of arguments
- `utils.py`: A file for defining utility functions
