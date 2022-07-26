import jax.numpy as np
import numpy as nnp

def unity(x):
    return x

def consensus(x,L : np.ndarray = None):
    if L is none:
        L = 
    return L @ x

class basic_system:
    def __init__(self,dimensions = 3, elements = 2):
        self.Gamma = unity
        self.H = unity
        self.x = np.zeros((elements, dimensions))
        self.f = consensus

class RO_SYS(basic_system):
