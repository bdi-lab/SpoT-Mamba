import numpy as np

class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean: float = mean
        self.std: float = std

    def fit_transform(self, data: np.ndarray):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray):
        return (data * self.std) + self.mean