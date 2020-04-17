import numpy as np


def linear_cost(X, y, theta, lamda):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (h - y) ** 2
    result1 = sq.sum() / (2 * m)
    tj = theta ** 2
    prueba = tj.sum() / (2 * m)
    result = result1 + (lamda * prueba) 
    return result
