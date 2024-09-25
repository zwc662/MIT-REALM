import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.controllers.pid import F16N0PIDController
from jax_f16.f16 import F16

from pncbf.dyn.f16_gcas import compute_f16_vel_angles
from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.angle_utils import wrap_to_pi
from pncbf.utils.jax_utils import jax_default_x32, jax_jit_np, jax_use_cpu, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()

    jax_default_x32()
    jax_use_cpu()
    set_logger_format()
    task = F16Two()

    tf = 40.0
    dt = task.dt / 2
    n_steps = int(round(tf / dt))

    # x0 = task.nominal_val_state()
    # fmt: off
    # x0 = np.array([ 3.1909497e+02,  5.9561551e-01,  2.8129080e-01, -4.3246555e-01,
    #     9.0162551e-01,  4.0905199e+00, -1.1320552e+00, -7.1275868e-03,
    #    -7.5986439e-01, -2.8705779e+03, -2.5670320e+03,  4.7344760e+02,
    #     9.9999969e+01,  1.1309022e+00, -6.3309807e-01, -2.4040976e-01,
    #     5.0772617e+02,  2.4952227e-02, -4.6277262e-09,  1.2113853e-07,
    #     2.5805691e-02, -2.3561945e+00, -1.8475431e-10, -3.2911648e-05,
    #     7.0306130e-09, -3.2832422e+03, -4.2832417e+03,  5.0251941e+02,
    #     9.0591068e+00,  2.9127996e-02,  1.8894505e-10, -2.7241057e-08])
    # x0 = np.array([ 4.1405246e+02, -2.5624748e-02, -1.0214587e-01, -6.2367625e+00,
    #    -4.1941866e-02,  2.3314912e+00, -1.9998224e+00,  1.7640223e-01,
    #     2.0295307e-01, -1.9569951e+03,  1.4482395e+03,  6.4351233e+02,
    #     9.9999969e+01, -1.1125168e-01,  1.2398719e+00,  5.8331415e-02,
    #     5.0772617e+02,  2.4952227e-02, -4.6277262e-09,  1.2113853e-07,
    #     2.5805691e-02, -2.3561945e+00, -1.8475431e-10, -3.2911648e-05,
    #     7.0306130e-09, -3.2832422e+03, -4.2832417e+03,  5.0251941e+02,
    #     9.0591068e+00,  2.9127996e-02,  1.8894505e-10, -2.7241057e-08])
    x0 = np.array([ 3.1850699e+02, -4.4939917e-02, -1.5636599e-01,  7.3925608e-01,
        3.0064970e-01,  2.4711277e+00, -7.8960866e-01, -4.1249874e-01,
       -3.0515915e-01, -1.0677603e+03,  3.8916913e+02,  3.5822693e+02,
        9.9999969e+01,  3.2605490e-01,  2.4116821e+00,  1.6646025e+00,
        5.0583084e+02,  2.5374869e-02, -5.5002802e-09,  1.2115248e-07,
        2.5944991e-02, -2.3561945e+00, -1.8826612e-10, -3.7466933e-05,
        7.2016619e-09, -1.8641559e+03, -2.8641553e+03,  5.0108731e+02,
        9.0586681e+00,  2.7471244e-02,  2.2251606e-10, -3.1282362e-08])
    # fmt: on

    sim = SimCtsPbar(
        task,
        task.nom_pol_N0_pid,
        n_steps,
        dt,
        dt0=dt,
        use_obs=False,
        use_pid=False,
        max_steps=n_steps,
        solver="bosh3",
        n_updates=2,
    )
    T_x, T_t = jax_jit_np(sim.rollout_plot)(x0)

    # Plot on pos2d. Put markers every
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], color="C1", marker="o", markevery=40, lw=0.5, alpha=0.9, zorder=8)
    # task.plot_pos2d(ax)
    ax.set(xlabel="East (ft)", ylabel="North (ft)", aspect="equal")
    fig.savefig(plot_dir / f"pos2d_nompid2.pdf", bbox_inches="tight")
    plt.close(fig)
    #######################################################################################
    # Also plot the PID terms.
    nom_alt = task.nominal_val_state()[F16.H]
    pid = F16N0PIDController(nom_alt)

    _, outs = jax_jit_np(jax_vmap(pid.get_control_all))(T_x[:, :F16.NX])

    #######################################################################################
    opts = dict(markevery=40, marker="o", lw=1.0, color="C1")

    nrows = task.nx + 4
    figsize = np.array([6, 1.0 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        ax.plot(T_t, T_x[:, ii], **opts)
        ax.set_ylabel(task.x_labels[ii], rotation=0, ha="right")

    axes[task.nx].plot(T_t, outs["pn_err"], **opts)

    axes[task.nx + 1].plot(T_t, outs["vn"], **dict(opts, color="C2"), label="vn")
    axes[task.nx + 1].plot(T_t, outs["vn_cmd"], **dict(opts, color="C3"), label="cmd")
    axes[task.nx + 1].plot(T_t, outs["vn_err"], **dict(opts, color="C4"), label="err")
    axes[task.nx + 1].set_ylabel("N vel", rotation=0, ha="right")
    axes[task.nx + 1].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    axes[task.nx + 2].plot(T_t, outs["psi_offset_pn_p"], **dict(opts, color="C2"), label="p")
    axes[task.nx + 2].plot(T_t, outs["psi_offset_pn_d"], **dict(opts, color="C3"), label="d")
    axes[task.nx + 2].plot(T_t, outs["psi_offset_pn"], **dict(opts, color="C4"), label="total")
    # axes[task.nx + 2].plot(T_t, outs["psi_cmd"], **dict(opts, color="C5"), label="psi")
    axes[task.nx + 2].set_ylabel("psi cmd", rotation=0, ha="right")
    axes[task.nx + 2].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    axes[task.nx + 3].plot(T_t, outs["phi_cmd"], **opts)
    axes[task.nx + 3].set_ylabel("phi cmd", rotation=0, ha="right")

    fig.savefig(plot_dir / f"pos2d_traj_nompid2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
