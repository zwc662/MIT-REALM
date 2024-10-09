import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import BoundaryNorm, CenteredNorm, Normalize

from clfrl.dyn.f16_gcas import A_BOUNDS, F16GCAS
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_double, jax_vmap, merge01, rep_vmap, unmerge01
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    plot_dir = get_script_plot_dir()
    task = F16GCAS()

    pol = task.nom_pol_pid
    n_pts = 100

    tf = 5.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    bT_x_safe = []

    b_x0s = []

    for setup in task.phase2d_setups():
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup.setup_idx, n_pts)
        b_x0 = merge01(bb_x)
        b_x0s.append(b_x0)

    # Also randomly sample from train state.
    key = jr.PRNGKey(124921)
    b_x0 = task.sample_train_x0(key, len(b_x0))
    b_x0s.append(b_x0)

    for b_x in b_x0s:
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        bT_x, bT_t = jax2np(jax_jit(jax_vmap(sim.rollout_plot))(b_x))
        bTh_h = jax2np(jax_jit(rep_vmap(task.h_components, rep=2))(bT_x))
        bh_Vh = np.max(bTh_h, axis=1)
        b_Vh = np.max(bh_Vh, axis=1)

        # Only keep the safe ones.
        b_is_safe = b_Vh < 0
        bT_x_safe.append(bT_x[b_is_safe, :])

        T_t = bT_t[0]
    bT_x_safe = np.concatenate(bT_x_safe, axis=0)

    # Compute the observations.
    bT_obs, _ = jax_jit(rep_vmap(task.get_obs, rep=2))(bT_x_safe)

    # Save results.
    npz_path = plot_dir / "obs_mean.npz"
    np.savez(npz_path, T_t=T_t, bT_obs=bT_obs, bT_x_safe=bT_x_safe)

    plot()


@app.command()
def plot():
    plot_dir = get_script_plot_dir()
    task = F16GCAS()

    npz_path = plot_dir / "obs_mean.npz"
    npz = EasyNpz(npz_path)

    T_t, bT_obs, bT_x_safe = npz("T_t", "bT_obs", "bT_x_safe")

    b_obs = merge01(bT_obs)
    b_x = merge01(bT_x_safe)

    np.set_printoptions(precision=2, linewidth=300, sign=" ")

    # Print min and max over states,
    train_bounds = task.train_bounds()
    print("----------------------------")
    print("States:")
    print("min: ", b_x.min(0))
    print("   : ", b_x.min(0) < train_bounds[0])
    print("max: ", b_x.max(0))
    print("   : ", b_x.max(0) > train_bounds[1])

    # Print min, median, max and quantiles.,
    print("----------------------------")
    q05 = np.quantile(b_obs, 0.01, axis=0)
    q95 = np.quantile(b_obs, 0.99, axis=0)
    print("Obs:")
    print("min: ", repr(b_obs.min(0)))
    print("q05: ", repr(q05))
    print("q50: ", repr(np.quantile(b_obs, 0.50, axis=0)))
    print("q95: ", repr(q95))
    print("max: ", repr(b_obs.max(0)))
    print("dif: ", repr(b_obs.max(0) - b_obs.min(0)))
    print("dif: ", repr(q95 - q05))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
