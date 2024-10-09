import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from clfrl.utils.jax_types import Arr


def plot_boundaries(axes: list[plt.Axes], bounds: Float[Arr, "2 nx"], color="C3", alpha=0.25):
    for ii, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        lb, ub = bounds[:, ii]

        if ymin < lb:
            ax.axhspan(ymin, lb, color=color, alpha=alpha)
        if ub < ymax:
            ax.axhspan(ub, ymax, color=color, alpha=alpha)
