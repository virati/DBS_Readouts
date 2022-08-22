import jax.numpy as np
from jax.numpy.linalg import norm
import numpy as nnp
from dbread.utils.functions import statics


class basic_system:
    def __init__(
        self,
        B: int = 1,
        M: int = 1,
        regions=10,
        dimensions=1,
    ):
        self.gamma = np.ones((regions, B))
        self.H = np.ones((regions, M))
        self.x = np.zeros((regions, dimensions))
        self.f = statics

        self.B = B
        self.M = M
        self.regions = regions
        self.dimensions = dimensions


class RO_SYS(basic_system):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def coverage(self) -> float:
        return np.dot(self.gamma.T, self.H) / (norm(self.gamma) * norm(self.H))
