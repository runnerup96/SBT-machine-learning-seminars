import numpy as np


def sigmoida(x):
    return 1. / (1 + np.exp(- x))

def softmax(x):
    e = np.exp(x) 
    return e / e.sum(axis=0)