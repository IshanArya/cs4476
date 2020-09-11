import numpy as np


def random_dice(N):
    return np.ceil(np.random.rand(N) * 6)

def reshape_vector(y):
    return y.reshape((3,2))

def max_value(z):
    maxValue = np.max(z, keepdims=True)
    x, y = np.where(z == maxValue)
    return x[0], y[0]

def count_ones(v):
    return len(v[v == 1])
