import numpy as np

def combination(W, a, b):
    return np.dot(W, a) +  b

def backward(dz, W, a0, b):
    # should return new W and b
    m = a0.shape[1]
    dW = (1./m) * np.dot(dz, a0.T)
    db = (1./m) * np.sum(dz)
    dA = np.dot(W.T, dz)
    return dW, db, dA