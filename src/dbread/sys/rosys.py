from .systems import base_system
import numpy as nnp
from sklearn.linear_model import LinearRegression
from typing import Any
from typing_extensions import Self
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
import scipy.stats as stats


class rosys(base_system):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_synth_states(self, T: int = 10_000, store=True) -> nnp.ndarray:
        x_states = super().H_oracle(T)

        if store:
            self.x_states = x_states
            return self
        else:
            return x_states

    def set_H(self, H_function: Callable = None, H_coeffs: nnp.ndarray = None, H_clip: int = 0) -> Self:
        if H_function is not None:
            self._H_function = H_function
        if H_coeffs is not None:
            self._H_coeffs = H_coeffs
        self._H_clip = H_clip

        return self

    def set_Γ(self, Γ_function: Callable, Γ_coeffs: nnp.ndarray) -> Self:
        self._Γ_function = Γ_function
        self._Γ_coeffs = Γ_coeffs

        return self

    def measure(self, x_states=None, plot=False, store=True) -> Self:
        if x_states is None:
            x_states = self.x_states
        y = self.H(x_states)
        if plot:
            plt.plot(y)
            plt.title('Measurement')
            plt.show()

        if store:
            self.y = y
            return self
        else:
            return y

    def behave(self, x_states=None, plot=False, store=True) -> Self:
        if x_states is None:
            x_states = self.x_states

        β = self.Γ(x_states)

        if plot:
            plt.plot(β)
            plt.title('Behavior')
            plt.show

        if store:
            self.β = β
            return self
        else:
            return β

    def train_readout(self) -> Self:
        self.Θ = LinearRegression().fit(self.y, self.β)

        return self

    def test_readout(self, x_test=None) -> Self:
        if x_test is None:
            x_test = self.gen_synth_states(
                T=1000, store=False)

        y_test = self.measure(x_test, store=False)
        β_hat = self.Θ.predict(y_test).squeeze()
        β_true = self.behave(x_states=x_test, store=False).squeeze()

        coeff = self.Θ.coef_.squeeze()
        gamma_coeffs = self._Γ_coeffs.squeeze()
        correlation = stats.pearsonr(β_hat, β_true)
        model_alignment = (np.dot(coeff, gamma_coeffs) /
                           (np.linalg.norm(gamma_coeffs)*np.linalg.norm(coeff)))

        return correlation, model_alignment
