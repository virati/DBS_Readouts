from .systems import base_system
import numpy as nnp
from sklearn.linear_model import LinearRegression
from typing import Any
from typing_extensions import Self
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np


class rosys(base_system):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_synth_states(self, T: int = 10_000, iso=False) -> nnp.ndarray:
        x_states = super().H_oracle(T)

        if iso:
            return x_states
        else:
            self.x_states = x_states
            return self

    def set_H(self, H_function: Callable, H_coeffs: nnp.ndarray) -> Self:
        self._H_function = H_function
        self._H_coeffs = H_coeffs

        return self

    def set_Γ(self, Γ_function: Callable, Γ_coeffs: nnp.ndarray) -> Self:
        self._Γ_function = Γ_function
        self._Γ_coeffs = Γ_coeffs

        return self

    def measure(self, plot=False) -> Self:
        self.y = self.H(self.x_states)
        if plot:
            plt.plot(self.y)
            plt.title('Measurement')
            plt.show()
        return self

    def behave(self, plot=False) -> Self:
        self.β = self.Γ(self.x_states)
        if plot:
            plt.plot(self.β)
            plt.title('Behavior')
            plt.show
        return self

    def train_readout(self) -> Self:
        self.Θ = LinearRegression().fit(self.y, self.β)

        return self

    def test_readout(self, y_test=None) -> Self:
        if y_test is None:
            y_test = self.y
        self.β_hat = self.Θ.predict(self.y)

        transformed_y = y_test * self.Θ.coef_[0]
        correlation_transformed = np.corrcoef()[0, 1]

        return self
