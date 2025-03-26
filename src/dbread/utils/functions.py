import numpy as np
from typing import Any


def unity(x: Any) -> Any:
    """
    Basic unity function
    """
    return x


def zero(x: Any) -> Any:
    return 0 * x


def stable(x: Any, c=0,) -> Any:
    """
    Stable dynamics in each dimensions at \vec{c} Γ
    """
    return -(x - c)


def all_to_all(x: np.ndarray) -> np.ndarray:
    """
    Basic all-to-all consensus dynamics
    """
    N = x.shape()[-1]
    L = np.ones(N, N)

    return L @ x
