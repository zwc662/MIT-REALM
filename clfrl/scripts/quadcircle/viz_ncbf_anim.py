import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import typer
from loguru import logger

import run_config.ncbf.quadcircle_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
import scripts.quadcircle.viz_intavoid_anim as intavoid_anim
from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from clfrl.ncbf.ncbf import NCBF
from clfrl.qp.min_norm_cbf import min_norm_cbf
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.ncbf.quadcircle_cfg.get(seed)
    nom_pol = task.nom_pol_vf
    alg: NCBF = NCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    x0 = task.get_state()

    tf = 15.0
    result_dt = 0.05
    alpha = 10.0
    pol = ft.partial(alg.get_cbf_control, alpha)
    sim = SimCtsReal(task, pol, tf, result_dt, use_pid=False, max_steps=384)
    logger.info("Rollout...")
    T_states, T_t, _ = jax2np(jax_jit(sim.rollout_plot)(x0))

    logger.info("Computing controls... Also show nominal policy, and why it is not safe...?")

    def get_control_info(state):
        u_lb, u_ub = task.u_min, task.u_max
        Vh = alg.get_V(state)
        x_Vx = jax.jacobian(alg.get_V)(state)
        f, G = task.f(state), task.G(state)

        u_nom = task.nom_pol_vf(state)
        # u_qp, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_Vh, hx_Vx, f, G, u_nom)
        u_qp, r, sol = min_norm_cbf(alpha, u_lb, u_ub, Vh, x_Vx, f, G, u_nom)
        u_qp = u_qp.clip(-1, 1)

        # Compute the constraint for both u_nom and u_qp.
        xdot_qp = f + G @ u_qp
        xdot_nom = f + G @ u_nom

        Vdot_qp = x_Vx @ xdot_qp
        Vdot_nom = x_Vx @ xdot_nom

        constr_rhs = -alpha * Vh

        return u_nom, u_qp, Vdot_nom, Vdot_qp, constr_rhs, r

    # T_controls = jax2np(jax_jit(jax_vmap(pol))(T_states))
    Tu_nom, Tu_qp, T_Vdot_nom, T_Vdot_qp, T_constr_rhs, T_r = jax2np(jax_jit(jax_vmap(get_control_info))(T_states))
    data = dict(
        Tu_nom=Tu_nom, Tu_qp=Tu_qp, T_Vdot_nom=T_Vdot_nom, T_Vdot_qp=T_Vdot_qp, T_constr_rhs=T_constr_rhs, T_r=T_r
    )

    Th_h = jax2np(jax_jit(jax_vmap(task.h_components))(T_states))
    T_Vh = jax2np(jax_jit(jax_vmap(alg.get_V))(T_states))
    Th_Vh_disc = jax_jit(ft.partial(compute_all_disc_avoid_terms, alg.lam, result_dt))(Th_h).Th_max_lhs

    npz_path = plot_dir / f"animdata{cid}.npz"
    np.savez(npz_path, T_states=T_states, Th_h=Th_h, T_Vh=T_Vh, Th_Vh_disc=Th_Vh_disc, T_t=T_t, **data)

    intavoid_anim.anim(ckpt_path)


@app.command()
def anim(ckpt_path: pathlib.Path, time: float = 15.0):
    intavoid_anim.anim(ckpt_path, time)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
