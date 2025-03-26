from typing import Union
from numpy import ndarray
import matplotlib.pyplot as plt
from dbread.sys.rosys import rosys


def plot_against_Γ(putative: rosys, to_plot: Union[ndarray, dict], label=None):
    if isinstance(to_plot, dict):
        label = to_plot.keys()[0]
        to_plot = to_plot[label]

    fig, ax1 = plt.subplots()
    ax1.plot(to_plot, 'r', label="")
    ax1.spines['left'].set_color('red')
    ax1.tick_params(axis='y', colors='red')
    plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(putative._Γ_coeffs[::-1], 'g--', label="Gamma Coeffs")
    ax2.spines['right'].set_color('green')
    ax2.tick_params(axis='y', colors='green')
    plt.title(label)
    plt.legend()
    plt.show()
