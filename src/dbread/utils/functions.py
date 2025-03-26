import numpy as np
from typing import Any, Optional, Tuple, Union


def innerprod(x: Any, y: Any) -> Any:
    """
    Inner product function
    """
    return np.inner(x, y.T)


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


def grad_nonzero_anywhere(x: np.ndarray, y: np.ndarray = None, domain: Optional[Union[Tuple[int], Tuple[float]]] = None, noise_threshold=1e-3, count_threshold=5) -> np.ndarray:
    """
    Check if gradient is non-zero anywhere
    """
    x_diff = np.diff(x, axis=-1)
    x_diff_where = np.where(np.abs(x_diff) > noise_threshold)
    if domain is None:
        x_diff_count = np.count_nonzero(x_diff_where)

    return x_diff_count > count_threshold, x_diff_count, x_diff_where
