import functools as ft
import pickle

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import Normalize

from clfrl.dyn.f16_gcas import F16GCAS
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from clfrl.plotting.contour_utils import centered_norm
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir
from clfrl.utils.schedules import lam_to_horizon

app = typer.Typer()


@app.command()
def gen():
    plot_dir = mkdir(get_script_plot_dir() / "disc_ci")
    task = F16GCAS()

    tf = 10.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    # L_lambds = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0])
    L_lambds = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    x0 = task.nominal_val_state()
    x0[task.H] = 450.0
    x0[task.THETA] = 0.1

    data = []
    setups = task.phase2d_setups()
    for ss, setup in enumerate(setups):
        logger.info("Plotting {}...".format(setup.plot_name))
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=ss, n_pts=96)

        def get_Th(state0):
            # sim = SimCtsReal(task, task.nom_pol_pid, tf, result_dt)
            sim = SimCtsPbar(task, task.nom_pol_pid, n_steps, dt, max_steps=n_steps, use_pid=False, solver="bosh3")
            T_x, T_t = sim.rollout_plot(state0)
            Th_h = jax_vmap(task.h_components)(T_x)
            return Th_h, T_x, T_t

        Th_h, T_x, T_t = get_Th(x0)
        bbTh_h, bbT_x, _ = jax2np(jax_jit(rep_vmap(get_Th, rep=2))(bb_x))

        # Plot how many timesteps before constraint violation.
        def get_t_violate(T_h):
            assert T_h.ndim == 1
            has_violate = jnp.any(T_h > 0.0)
            t_violate = jnp.argmax(T_h >= 0.0)
            return jnp.where(has_violate, t_violate, jnp.inf) * dt

        bbh_t_viol = jax2np(jax_jit(rep_vmap(jax_vmap(get_t_violate, in_axes=1), rep=2))(bbTh_h))

        def get_Vh_disc(lambd):
            bb_terms = rep_vmap(ft.partial(compute_all_disc_avoid_terms, lambd, dt), rep=2)(bbTh_h)
            bbVh_h = bb_terms.Th_max_lhs[:, :, 0]

            return bbVh_h

        Lbbh_Vh = jax2np(jax_jit(jax_vmap(get_Vh_disc))(L_lambds))
        Lbb1_Vhmax = np.max(Lbbh_Vh, axis=3, keepdims=True)
        Lbbh_Vh = np.concatenate([Lbbh_Vh, Lbb1_Vhmax], axis=3)

        data.append([bb_Xs, bb_Ys, T_x, bbh_t_viol, Lbbh_Vh])

    pkl_path = plot_dir / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump([L_lambds, data], f)
    plot()


@app.command()
def plot():
    plot_dir = mkdir(get_script_plot_dir() / "disc_ci")
    task = F16GCAS()

    pkl_path = plot_dir / "data.pkl"
    with open(pkl_path, "rb") as f:
        L_lambds, data = pickle.load(f)

    n_lambds = len(L_lambds)

    setups = task.phase2d_setups()
    for ss, setup in enumerate(setups):
        bb_Xs, bb_Ys, T_x, bbh_t_viol, Lbbh_Vh = data[ss]

        bb1_t_viol_min = np.min(bbh_t_viol, axis=2, keepdims=True)
        bbh_t_viol = np.concatenate([bbh_t_viol, bb1_t_viol_min], axis=2)
        bbh_t_viol = np.ma.array(bbh_t_viol, mask=~np.isfinite(bbh_t_viol))

        t_viol_min = bb1_t_viol_min.flatten()
        t_viol_min = t_viol_min[np.isfinite(t_viol_min)]
        ############################################################################################
        fig, ax = plt.subplots(layout="constrained")
        ax.hist(t_viol_min, bins=48)
        fig.savefig(plot_dir / f"time_to_violate_dist_{setup.plot_name}.pdf")
        plt.close(fig)

        ############################################################################################
        norm = Normalize(vmin=0, vmax=1.005 * t_viol_min.max())
        figsize = (2 * (task.nh + 1), 2)
        fig, axes = plt.subplots(1, task.nh + 1, figsize=figsize, layout="constrained")
        for ii, ax in enumerate(axes):
            cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_t_viol[:, :, ii], norm=norm, levels=15, cmap="rocket_r")
            setup.plot(ax)
            if ii < task.nh:
                ax.set_title(task.h_labels[ii])
            else:
                ax.set_title("max: {:.1f}".format(bbh_t_viol[:, :, ii].max()))
        fig.colorbar(cs0, ax=axes.ravel().tolist(), shrink=0.95)
        fig.savefig(plot_dir / f"time_to_violate_{setup.plot_name}.pdf")
        plt.close(fig)
        ############################################################################################
        # Use same norm for all discounts for ecah h.
        norms = [centered_norm(Lbbh_Vh[:, :, :, jj].min(), Lbbh_Vh[:, :, :, jj].max()) for jj in range(task.nh + 1)]
        levels = [np.linspace(-norm.halfrange, norm.halfrange, num=25) for norm in norms]

        ax: plt.Axes
        figsize = (3 * n_lambds, 2 * (task.nh + 1))
        fig, axes = plt.subplots(task.nh + 1, n_lambds, figsize=figsize, layout="constrained")

        for ii in range(n_lambds):
            for jj, ax in enumerate(axes[:, ii]):
                ax.plot(T_x[:, setup.idx0], T_x[:, setup.idx1], "C1", linewidth=0.3, zorder=10)
                setup.plot(ax)
                cs0 = axes[jj, ii].contourf(
                    bb_Xs,
                    bb_Ys,
                    Lbbh_Vh[ii, :, :, jj],
                    levels=levels[jj],
                    norm=norms[jj],
                    cmap="RdBu_r",
                    zorder=3.5,
                    alpha=0.9,
                )
                cs1 = axes[jj, ii].contour(
                    bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=[0.0], linewidths=[1.0], colors=["C5"], zorder=10.0
                )

                if ii > 0:
                    # Plot the true one.
                    axes[jj, ii].contour(
                        bb_Xs,
                        bb_Ys,
                        Lbbh_Vh[0, :, :, jj],
                        levels=[0.0],
                        colors=["C2"],
                        linestyles=["--"],
                        alpha=0.7,
                        zorder=8.0,
                    )
                    # Plot the shifted discounted one.
                    shift = (1 - np.exp(-L_lambds[ii] * task.max_ttc)) * task.h_min
                    axes[jj, ii].contour(
                        bb_Xs,
                        bb_Ys,
                        Lbbh_Vh[ii, :, :, jj] - shift,
                        levels=[0.0],
                        colors=["C4"],
                        linestyles=["--"],
                        alpha=0.7,
                        zorder=8.0,
                    )

                if ii == 0:
                    # Colorbar for each h.
                    cbar = fig.colorbar(cs0, ax=axes[jj, :].ravel().tolist())
                    cbar.add_lines(cs1)
            lam_T = lam_to_horizon(L_lambds[ii], task.dt)
            axes[0, ii].set_title(rf"$\lambda={L_lambds[ii]}$" + "\nT={}".format(lam_T))
        # Overwrite the ylabels.
        h_labels = task.h_labels + ["max"]
        for r, ax in zip(h_labels, axes[:, 0]):
            ax.set_ylabel(r, rotation=0, size="large")

        fig.savefig(plot_dir / f"check_discr_{setup.plot_name}.pdf")
        ############################################################################################


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()