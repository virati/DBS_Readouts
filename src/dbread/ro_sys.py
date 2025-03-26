import numpy as nnp
from dbread.utils.functions import statics
import matplotlib.pyplot as plt
import networkx as nx


class base_system:
    def __init__(
        self,
        num_nodes: int = 1,
        num_probes: int = 1,
        num_behaviors: int = 1,
    ):
        self.num_nodes = num_nodes
        self.num_probes = num_probes
        self.num_behaviors = num_behaviors

        self._H_coeffs = nnp.zeros((num_nodes, num_probes))
        self._Γ_coeffs = nnp.zeros((num_nodes, num_behaviors))

        self._x_graph = nx.gnp_random_graph(num_nodes, 0.5)

    def plot_coverage(self):
        plt.figure()
        plt.imshow([self._H_coeffs, self._Γ_coeffs],
                   cmap="hot", interpolation="None")
        plt.show()

    def plot_x_space(self):
        plt.figure()
        nx.draw(self._x_graph, with_labels=True)
        plt.title('Ground Truth Connectivity in x-layer')

    def get_x_timeseries(self, T: int = 1000):
        covariance_matrix = nx.laplacian_matrix(self._x_graph)

        x_timeseries = nnp.random.multivariate_normal(
            nnp.zeros(self.num_nodes), covariance_matrix.todense(), T)
        return x_timeseries


class RO_SYS(base_system):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def coverage(self) -> float:
        return nnp.dot(self.gamma.T, self.H) / (nnp.sum(self.gamma))

    def gen_synth_states(self, T: int = 10_000) -> nnp.ndarray:
        self.X_states = nnp.random.multivariate_normal(
            0 * nnp.zeros(self.regions), nnp.eye(self.regions), size=(T,)
        )

        return self

    def measure(self):
        return nnp.dot(base_system.H.T, self.X_states.T).squeeze()

    def behave(self):
        return nnp.dot(base_system.gamma.T, self.X_states.T).squeeze()

    def prediction_stats(self, plot=False, X=None, Y=None):
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.scatter(X, Y)
            pearsonr(Y, X)

            model = sm.OLS(Y, X)
