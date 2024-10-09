import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from clfrl.dyn.f16_gcas import F16GCAS
from clfrl.utils.jax_utils import jax2np, jax_use_cpu, jax_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    jax_use_cpu()

    plot_dir = mkdir(get_script_plot_dir() / "viz_h")
    task = F16GCAS()

    batch_size = 256

    a_buf, b_buf = 5e-2, 5e-2
    a_buf2, b_buf2 = 5e-2, 5e-2

    # alpha.
    bounds = task.train_bounds()
    bounds_alpha, bounds_beta = bounds[:, task.ALPHA], bounds[:, task.BETA]
    b_alphas = np.linspace(bounds_alpha[0] - a_buf2, bounds_alpha[1] + a_buf2, num=batch_size)
    x0 = task.nominal_val_state()

    b_x0 = ei.repeat(x0, "nx -> b nx", b=batch_size)
    b_x0[:, task.ALPHA] = b_alphas
    bh_h_alpha = jax2np(jax_vmap(task.h_components)(b_x0))

    ###########
    b_betas = np.linspace(bounds_beta[0] - b_buf2, bounds_beta[1] + b_buf2, num=batch_size)
    b_x0 = ei.repeat(x0, "nx -> b nx", b=batch_size)
    b_x0[:, task.BETA] = b_betas
    bh_h_beta = jax2np(jax_vmap(task.h_components)(b_x0))

    ###########
    fig, axes = plt.subplots(2, layout="constrained")
    axes[0].plot(b_alphas, bh_h_alpha[:, 2], label=r"$\alpha_l$")
    axes[0].plot(b_alphas, bh_h_alpha[:, 3], label=r"$\alpha_u$")
    axes[1].plot(b_betas, bh_h_beta[:, 4], label=r"$\beta_l$")
    axes[1].plot(b_betas, bh_h_beta[:, 5], label=r"$\beta_u$")
    [ax.legend() for ax in axes]
    axes[0].set(xlabel=r"$\alpha$")
    axes[1].set(xlabel=r"$\beta$")

    axes[0].axvline(bounds_alpha[0], ls="--", color="C3")
    axes[0].axvline(bounds_alpha[0] + a_buf, ls="--", color="C4")
    axes[0].axvline(bounds_alpha[1], ls="--", color="C3")
    axes[0].axvline(bounds_alpha[1] - a_buf, ls="--", color="C4")

    axes[1].axvline(bounds_beta[0], ls="--", color="C3")
    axes[1].axvline(bounds_beta[1], ls="--", color="C3")

    fig.savefig(plot_dir / "h_alphabeta.pdf")
    plt.close(fig)
    ###########
    b_thetas = np.linspace(-np.pi / 2, np.pi / 2, num=batch_size)
    b_x0 = ei.repeat(x0, "nx -> b nx", b=batch_size)
    b_x0[:, task.THETA] = b_thetas
    bh_h_theta = jax2np(jax_vmap(task.h_components)(b_x0))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_thetas, bh_h_theta[:, 8], label=r"$\theta_l$")
    ax.plot(b_thetas, bh_h_theta[:, 9], label=r"$\theta_u$")
    ax.axvline(1.2, ls="--", color="C3")
    ax.axvline(-1.2, ls="--", color="C3")
    ax.set(xlabel=r"$\theta$")
    fig.savefig(plot_dir / "h_theta.pdf")
    plt.close(fig)
    ###########
    b_nyr = np.linspace(-3.0, 3.0, num=batch_size)
    b_x0 = ei.repeat(x0, "nx -> b nx", b=batch_size)
    b_x0[:, task.NYRINT] = b_nyr
    bh_h_nyr = jax2np(jax_vmap(task.h_components)(b_x0))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_nyr, bh_h_nyr[:, 6], label=r"$nyr_l$")
    ax.plot(b_nyr, bh_h_nyr[:, 7], label=r"$nyr_u$")
    ax.axvline(2.5, ls="--", color="C3")
    ax.axvline(-2.5, ls="--", color="C3")
    ax.set(xlabel=r"$ny+R$")
    fig.savefig(plot_dir / "h_nyrint.pdf")
    plt.close(fig)
    ###########
    b_alt = np.linspace(-100.0, 800.0, num=batch_size)
    b_x0 = ei.repeat(x0, "nx -> b nx", b=batch_size)
    b_x0[:, task.H] = b_alt
    bh_h_alt = jax2np(jax_vmap(task.h_components)(b_x0))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_alt, bh_h_alt[:, 0], label=r"$h_l$")
    ax.plot(b_alt, bh_h_alt[:, 1], label=r"$h_u$")
    ax.axvline(0.0, ls="--", color="C3")
    ax.axvline(task._alt_max, ls="--", color="C3")
    ax.set(xlabel=r"alt")
    fig.savefig(plot_dir / "h_alt.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
