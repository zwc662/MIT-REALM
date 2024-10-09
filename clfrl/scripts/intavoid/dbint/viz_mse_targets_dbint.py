import functools as ft

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.ncbf.compute_disc_avoid import AllDiscAvoidTerms, compute_all_disc_avoid_terms
from clfrl.utils.jax_utils import jax_jit, rep_vmap
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    task = DoubleIntWall()
    nom_pol = task.nom_pol_osc

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

    lams = [0.1, 0.5, 1.0]
    n_lams = len(lams)

    T = 80
    tf = T * task.dt
    rollout_dt = 0.5 * task.dt
    sim = SimCtsReal(task, nom_pol, tf, rollout_dt, use_obs=False)
    bbT_x, _, _ = jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x)
    bbTh_h = rep_vmap(task.h_components, rep=3)(bbT_x)
    bb_vterms: list[AllDiscAvoidTerms] = [
        rep_vmap(ft.partial(compute_all_disc_avoid_terms, lam, rollout_dt), rep=2)(bbTh_h) for lam in lams
    ]

    figsize = np.array([2 * task.nh, 2 * n_lams])
    fig, axes = plt.subplots(n_lams, task.nh, figsize=figsize, layout="constrained")
    for ii, lam in enumerate(lams):
        for jj, ax in enumerate(axes[ii, :]):
            ax.contourf(
                bb_Xs, bb_Ys, bb_vterms[ii].Th_max_lhs[:, :, 0, jj], levels=31, cmap="RdBu_r", norm=CenteredNorm()
            )
            task.plot_phase(ax)
    [axes[0, jj].set_title(task.h_labels[jj]) for jj in range(task.nh)]
    fig.savefig(plot_dir / "dbint_vterms.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
