import pathlib

import einops as ei
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from efppo.task.dyn_types import BTState, BState, BLFloat 
from efppo.task.task import Task
from efppo.utils.register_cmaps import register_cmaps


def C(ii: int):
    colors = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
    return colors[ii % len(colors)]


class Plotter:
    def __init__(self, task: Task, rel_path: pathlib.Path | None = None, dpi: int = None):
        self.task = task
        self.rel_path = rel_path
        self.dpi = dpi
        register_cmaps()

    def plot_traj(self, bT_state: BTState, multicolor: bool = False, ax: plt.Axes = None):
        bT_x, bT_y = self.task.get2d(bT_state)
        bT_line = np.stack([bT_x, bT_y], axis=-1)

        if ax is None:
            fig, ax = plt.subplots(dpi=self.dpi)
        else:
            fig = ax.figure

        self.task.setup_traj_plot(ax)
        colors = C(1)
        if multicolor:
            colors = [C(ii) for ii in range(bT_line.shape[1])]
        #line_col = LineCollection(bT_line, lw=1.0, zorder=5, colors=colors)
        #ax.add_collection(line_col)

        for i in range(bT_x.shape[0]):
            ax.scatter(bT_x[i, 1:-1], bT_y[i, 1:-1], color = colors[i], zorder=5, s=1**2)

        # Starts and Ends.
        ax.scatter(bT_x[:, 0], bT_y[:, 0], color="black", s=1**2, zorder=6, marker="s")
        ax.scatter(bT_x[:, -1], bT_y[:, -1], color="green", s=1**2, zorder=7, marker="o")

        ax.autoscale_view()
        return fig

    def plot_traj2(self, bT_x):
        figsize = 1.5 * np.array([8, 2 * self.task.nx])

        b, T, _ = bT_x.shape
        T_t = np.arange(T)
        bT_t = ei.repeat(T_t, "T -> b T", b=b)

        fig, axes = plt.subplots(self.task.nx, figsize=figsize, sharex=True, layout="constrained")
        for ii, ax in enumerate(axes):
            bT_xi = bT_x[:, :, ii]
            bT_line = np.stack([bT_t, bT_xi], axis=-1)
            colors = [C(ii) for ii in range(bT_line.shape[1])]
            line_col = LineCollection(bT_line, lw=1.0, zorder=5, colors=colors)
            ax.add_collection(line_col)
            ax.autoscale_view()

            ax.set_ylabel(self.task.x_labels[ii])
        self.task.setup_traj2_plot(axes)
        return fig


    def plot_traj3(self, bT_x, bT_h, bT_l):
        figsize = 1.5 * np.array([8, 2 * (self.task.nx + 2)])

        b, T, _ = bT_x.shape
        T_t = np.arange(1, T)
        bT_t = ei.repeat(T_t, "T -> b T", b=b)

        assert b == bT_l.shape[0] == bT_h.shape[0], f"{b=} {bT_x.shape=} {bT_l.shape=} {bT_h.shape=}"
        assert T - 1 == bT_l.shape[1] == bT_h.shape[1], f"{b=} {T=} {bT_l.shape=} {bT_h.shape=}"
        

        fig, axes = plt.subplots(self.task.nx + 1 + len(self.task.h_labels), figsize=figsize, sharex=True, layout="constrained")
        

        for ii, ax in enumerate(axes):
            bT_line = []
            if ii < self.task.nx:
                bT_xi = bT_x[:, 1:, ii]
                bT_line = np.stack([bT_t, bT_xi], axis=-1)
                ax.set_ylabel(self.task.x_labels[ii])
            elif ii - self.task.nx == 0:
                bT_line = np.stack([bT_t, bT_l], axis=-1)
                ax.set_ylabel('l')
            elif ii - self.task.nx >= 1:
                bT_line = np.stack([bT_t, bT_h[..., ii - self.task.nx - 1]], axis=-1)
                ax.set_ylabel(f'h_{self.task.h_labels[ii - self.task.nx - 1]}')
            

            colors = [C(ii) for ii in range(bT_line.shape[1])]
            line_col = LineCollection(bT_line, lw=1.0, zorder=5, colors=colors)
            ax.add_collection(line_col)
            ax.autoscale_view()
           
                
        
           
        self.task.setup_traj2_plot(axes)
        return fig

    def plot_dots(self, states: BState, colors: BLFloat, ax: plt.Axes = None):
        xs, ys = self.task.get2d(states)
        colors = (colors - np.min(colors)) / (1e-2 + np.max(colors) - np.min(colors))
        if ax is None:
            fig, ax = plt.subplots(dpi=self.dpi)
        else:
            fig = ax.figure

        self.task.setup_traj_plot(ax)
         
        #line_col = LineCollection(bT_line, lw=1.0, zorder=5, colors=colors)
        #ax.add_collection(line_col)

        scatters = ax.scatter(xs, ys, c = colors, cmap='viridis', zorder=5, s=0.8**2)
        # Add a colorbar to show the heatmap scale
        plt.colorbar(scatters, ax=ax)

        ax.autoscale_view()
        return fig