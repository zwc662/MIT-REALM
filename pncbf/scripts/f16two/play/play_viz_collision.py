import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import CenteredNorm

from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit_np, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir
from pncbf.utils.paths import get_script_plot_dir
from pncbf.utils.sampling_utils import get_mesh_np

app = typer.Typer()


@app.command()
def gen():
    jax_default_x32()
    set_logger_format()
    plot_dir = mkdir(get_script_plot_dir() / "play_viz_collision")

    task = F16Two()

    x0 = task.nominal_val_state()
    x0[task.PE0] = -2_500
    x0[task.PN0] = -200
    x0[task.PE1] = 0.0
    x0[task.PN1] = 1_000.0
    # x0[task.PSI1] = -0.75 * np.pi
    x0[task.PSI1] = -0.85 * np.pi

    tf = 16.0
    dt = task.dt / 4
    n_steps = int(round(tf / dt))

    sim = SimCtsPbar(
        task, task.nom_pol_pid, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
    )
    T_x_nom, T_t_nom = jax_jit_np(sim.rollout_plot)(x0)
    Th_h_nom = jax2np(jax_vmap(task.h_components)(T_x_nom))

    # Find the idx where the collision is the closest.
    kk_col = Th_h_nom[:, -1].argmax()

    x_col = T_x_nom[kk_col]

    # Do a contour plot around plane0 pe-pn to visualize the h_col.
    n_pts = 64
    contour_bounds = task.contour_bounds()
    contour_bounds[:, task.PE0] = x_col[task.PE0] + np.array([-500.0, 500.0])
    contour_bounds[:, task.PN0] = x_col[task.PN0] + np.array([-500.0, 500.0])
    idxs = [task.PE0, task.PN0]
    bb_Xs, bb_Ys, bb_x0 = get_mesh_np(contour_bounds, idxs, n_pts, n_pts, x_col)

    bbh_h = jax_jit_np(jax_vmap(task.h_components, rep=2))(bb_x0)
    bb_h = bbh_h[:, :, -1]

    fig, ax = plt.subplots()
    # Marker on plane0 and plane1 pos.
    ax.plot(x_col[task.PE0], x_col[task.PN0], marker="o", mfc="C1", mec="C5", zorder=10)
    ax.plot(x_col[task.PE1], x_col[task.PN1], marker="o", mfc="C0", mec="C5", zorder=10)
    # Plot traj.
    ax.plot(T_x_nom[:, task.PE0], T_x_nom[:, task.PN0], color="C1", lw=0.5, alpha=0.5, zorder=6)
    ax.plot(T_x_nom[:, task.PE1], T_x_nom[:, task.PN1], color="C0", lw=0.5, alpha=0.5, zorder=6)

    # Visualize the heading of plane1.
    # psi1=0 => north (0, 1). psi1=0.5pi => east (1, 0).
    vec = np.array([np.sin(x_col[task.PSI1]), np.cos(x_col[task.PSI1])])
    pt0 = x_col[[task.PE1, task.PN1]]
    pt1 = pt0 + vec * 200.0
    seg = np.stack([pt0, pt1], axis=0)
    ax.plot(seg[:, 0], seg[:, 1], color="C0", lw=1.0, zorder=10)

    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_h, norm=CenteredNorm(), cmap="RdBu_r")
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_h, levels=[0.0], color="black")
    cbar = fig.colorbar(cs0, ax=ax)
    cbar.add_lines(cs1)
    ax.set_aspect("equal")
    ax.set(xlim=contour_bounds[:, task.PE0], ylim=contour_bounds[:, task.PN0])
    ax.set(xlabel="PE", ylabel="PN")
    fig.savefig(plot_dir / "h_col_nompolpid.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
