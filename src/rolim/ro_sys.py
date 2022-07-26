import jax.numpy as np
import numpy as nnp


def unity(x):
    """
    Basic unity function
    """
    return x


def statics(x):
    """
    Static dynamics
    """
    return 0


def stables(
    x,
    c=0,
):
    """
    Stable dynamics in each dimensions at \vec{c} Γ
    """
    return -(x - c)


def consensus(x):
    """
    Basic all-to-all consensus dynamics
    """
    N = x.shape()[-1]
    L = np.ones(N, N)

    return L @ x


class basic_system:
    def __init__(self, B=1, M=1, regions=2, dimensions=1):
        self.Γ = np.ones(regions, B)
        self.H = np.ones(regions, M)
        self.x = np.zeros((regions, dimensions))
        self.f = statics


class RO_SYS(basic_system):
    def __init__(self, **kwargs):
        super.__init__(self, **kwargs)

    def coverage(self) -> float:
        return self.Γ @ self.H / np.abs(self.Γ)
