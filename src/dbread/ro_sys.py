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
        return np.dot(self.gamma.T, self.H) / (np.sum(self.gamma))

    def gen_synth_states(self, T: int = 10_000) -> np.ndarray:
        self.X_states = nnp.random.multivariate_normal(
            0 * np.zeros(self.regions), nnp.eye(self.regions), size=(T,)
        )

        return self

    def measure(self):
        return nnp.dot(basic_system.H.T, self.X_states.T).squeeze()

    def behave(self):
        return nnp.dot(basic_system.gamma.T, self.X_states.T).squeeze()

    def prediction_stats(self, plot=False, X=None, Y=None):
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.scatter(X, Y)
            pearsonr(Y, X)

            model = sm.OLS(Y, X)
