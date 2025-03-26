import numpy as nnp
import matplotlib.pyplot as plt
import networkx as nx
from abc import abstractmethod, ABC
from typing import Any
from typing_extensions import Self


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
        self.x_states = None

    def H(self, input):
        return self._H_function(input, self._H_coeffs)

    def Γ(self, input):
        return self._Γ_function(input, self._Γ_coeffs)

    def plot_graph(self):
        plt.figure()
        nx.draw(self._x_graph, with_labels=True)
        plt.title('Ground Truth Connectivity in x-layer')
        plt.show()

    def plot_connectivity(self):
        plt.figure()
        plt.imshow(nx.laplacian_matrix(self._x_graph).todense(),
                   cmap="hot", interpolation="None")
        plt.show()

    def plot_coverage(self) -> None:
        plt.figure()
        plt.imshow([self._H_coeffs.max(axis=-1), self._Γ_coeffs.max(axis=-1)],
                   interpolation="None")
        plt.colorbar()
        plt.show()

    def plot_x(self):
        x = self.H_oracle() if self.x_states is None else self.x_states

        plt.figure()
        plt.plot(x, alpha=0.2)
        plt.title('X Timeseries')

    def H_oracle(self, T: int = 1000) -> nnp.ndarray:
        covariance_matrix = nx.laplacian_matrix(self._x_graph)

        x_timeseries = nnp.random.multivariate_normal(
            nnp.zeros(self.num_nodes), covariance_matrix.todense(), T)
        return x_timeseries
