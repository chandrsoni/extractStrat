import numpy as np

def sigmoid(h):
    return 1./(1. + np.exp(-h))