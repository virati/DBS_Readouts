# %%
"""
[Paper] Readout Limitations: RvA and Coverage Metrics
Summary: This notebook focuses on comparing RvA vs AvR, clarifying misconceptions about R^2 in readout assessment.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import logging

# %%
"""# Rational Assessment of Disease Readouts for Adaptive Deep Brain Stimulation

Efforts to engineer *biomarkers* of disease have grown rapidly in the last 10 years.
However, a framework to assess biomarkers, and more generally any physiologically-derived *readout* of a disorder, is lacking.

In this study, we take an *applied mathematical approach* to develop and demonstrate a framework that can be used to assess readouts more systematically.
This is a critical step in avoiding both overestimations and underestimations of readouts for adaptive DBS applications.


## Outline
* Re-referencings
* RvA vs AvR
* Fixed $\Gamma$ and Fixed H
* Fixed $\Gamma$ $\rightarrow$ Design H

## Re-Referencings

## RvA vs AvR
"""

# %%


def simple_H(x, c): return np.dot(c.T, x)
def simple_Gamma(x, c): return np.dot(c.T, x)


class readout:
    """
    A readout, at it's fundamental level, is a measurement of a brain...
    """

    def __init__(self, D=10, N=100, do_defaults=True):
        self.D = D
        self.N = N
        if do_defaults:
            self.set_H(simple_H, prune=3)
            self.set_Gamma(simple_Gamma, prune=3)

    def initialize_layers(self):
        D, N = self.D, self.N

        self.x = np.random.multivariate_normal(
            mean=np.zeros(D), cov=np.eye(D), size=N).T

        self.y = self.H(self.x, self.H_coverage)
        self.b = self.Gamma(self.x, self.gamma_coverage)

    def set_H(self, use_H, prune=0):
        self.H_coverage = np.ones((self.D, 1))
        self.H = use_H

        # prune readout
        if prune > 0:
            self.H_coverage[-prune:, ...] = 0

        # prune behavior

    def set_Gamma(self, use_Gamma, prune=0):
        self.gamma_coverage = np.ones((self.D, 1))
        self.Gamma = use_Gamma

        if prune > 0:
            self.gamma_coverage[:prune, ...] = 0


class assessment:
    def __init__(self, putatitve_ro: readout = None):
        if putatitve_ro is None:
            raise ValueError('No readout has been provided')
        self._regression = None
        self._putative_ro = putatitve_ro

    def regression(self, input, output):
        if input is None:
            input = self._putative_ro.y
        if output is None:
            output = self._putative_ro.b

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            input, output)
        self._regression = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'x-axis': input,
            'y-axis': output
        }

        return self

    def rb_coverage(self):
        measurement_brain_coverage = self._putative_ro.H_coverage
        behavior_brain_coverage = self._putative_ro.gamma_coverage

        plt.imshow([measurement_brain_coverage, behavior_brain_coverage])
        plt.yticks(ticks=[0, 1], labels=['Readout', 'Behavior'])
        plt.xlabel('Brain Node')
        plt.show()

    def print_regression(self, stats=None):
        if self._regression is None:
            raise ValueError("No Regression has been performed")

        if stats is None:
            stats = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']

        [print(f"{stat} {self._regression[stat]}") for stat in stats]
        return self

    def plot(self, input=None, output=None):
        if self._regression is None:
            logging.warning('No regression has been performed')
            if input is None or output is None:
                raise ValueError("No Data to Plot")
        else:
            input = self._regression['x-axis']
            output = self._regression['y-axis']

        plt.scatter(input, output)

        return self


# %% Run through the thing
perfect_ro = readout(do_defaults=True)
analysis_rvb = assessment(perfect_ro)

analysis_rvb.rb_coverage()
analysis_rvb.plot()

# %%


plt.scatter(y, b)
plt.xlabel('Readout')
plt.ylabel('Behavior')
plt.show()

# %%
# do a regression

# %%
rvb = assessment()
bvr = assessment()


result_rvb = rvb.regression(y, b).plot(
).print_regression(stats=['slope', 'r_value'])
plt.title('Readout vs Behavior')
plt.xlabel('Readout')
plt.ylabel('Behavior')
plt.show()

result_bvr = bvr.regression(b, y).plot(
).print_regression(stats=['slope', 'r_value'])
plt.title('Behavior vs Readout')
plt.xlabel('Behavior')
plt.ylabel('Readout')
plt.show()
