import numpy as np

def rbf_kernel(t):
    return np.exp( - 1/2 * t**2  )

