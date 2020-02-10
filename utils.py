"""
Utility module for conducting different operations on the dataset
"""

def normalize_2d(data, w, h):
    return (2.0*data/w) - [1, (h*1.0)/w]
