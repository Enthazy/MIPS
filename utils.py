import numpy as np
from numba import jit, njit

def cal_contact():
    pass

@njit
def FJPotential(x, y, d, r6=1e-3, epsilon=1e-3, **kwargs):
    t = np.sqrt(x**2 + y**2)
    rx = x/t
    ry = y/t
    s = epsilon * r6 * (d ** 6 - 2 * r6) / d ** 13
    return s*rx, s*ry

