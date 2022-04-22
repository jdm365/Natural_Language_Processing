import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
