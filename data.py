import numpy as np

def generateData(n=100, d=2, scale=10):
    #Defualt Dimension is 2.
    X = np.random.rand(n,d)
    X = scale * X
    Y = np.random.rand(n, 1)
    Y = scale * Y
    return X, Y