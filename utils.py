"""
Utility module for conducting different operations on the dataset
"""
import numpy as np

def normalize_2d(data, w, h):
    return (2.0*data/w) - [1, (h*1.0)/w]

def read_npz(file_path):
    data = np.load(file_path, allow_pickle=True)['data']

    return data.item()
