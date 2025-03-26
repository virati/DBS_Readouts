from .systems import base_system
import numpy as nnp
from sklearn.linear_model import LinearRegression
from typing import Self, Any


class rosys(base_system):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_synth_states(self, T: int = 10_000, iso=False) -> nnp.ndarray:
        x_states = super().get_x_timeseries(T)

        if iso:
            return x_states
        else:
            self.x_states = x_states
            return self

    def measure(self) -> Self:
        self.y = self.H(self.X_states)

        return self

    def behave(self):
        self.β = self.Γ(self.X_states)

        return self

    def train_readout(self):
        self.Θ = LinearRegression().fit(self.y, self.β)

        return self

    def test_readout(self, y_test=None):
        if y_test is None:
            y_test = self.y
        self.β_hat = self.Θ.predict(self.y)

        return self
