import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def grid(grid_size, grid_interval):
    """
    Constructs a 3d image of a grid with a given size
    For visualization purposes
    Args:
        grid_size: (x, y, z)
        grid_interval: The interval between the grid lines
    """
    image = np.zeros(grid_size)
    # Draw vertical lines
    image[::grid_interval, :, :] = 1
    image[:, ::grid_interval, :] = 1
    image[:, :, ::grid_interval] = 1
    return image


class RunningMeanVar():
    """
    This class is used to calculate the mean and variance of the n last values
    Example:
        >>> rmv = RunningMeanVar(n=10)
        >>> for i in range(100):
        >>>     rmv.add(i)
        >>>     print(rmv.mean())
        
    Args:
        n: The number of values to use for calculating the mean and variance
    """

    def __init__(self, n=10):
        self.n = n
        self.values = np.zeros(n)
        self.last_index = 0

    def add(self, value):
        """ Add a value to the running mean and variance
        Args:
            value: The value to add
        
        """
        self.values[self.last_index] = value
        self.last_index = (self.last_index + 1) % self.n

    def mean(self):
        """
        Returns: The mean of the last n values
        """
        return np.mean(self.values)

    def var(self):
        """
        Returns: The variance of the last n values
        """
        return np.var(self.values)


def VectorPearsonCorrelation(x, y):
    """
    Calculates the Pearson correlation between two vectors sets
    Args:
        x: (num_samples, num_features)
        y: (num_samples, num_features)
    """
    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    sigma_x = torch.sqrt(torch.sum(diff_x**2))
    sigma_y = torch.sqrt(torch.sum(diff_y**2))

    pcc = torch.sum(diff_x * diff_y) / (sigma_x * sigma_y)
    return pcc
