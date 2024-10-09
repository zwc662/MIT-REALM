import ipdb
import jax

from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.utils.jax_utils import jax2np, jax_jit, rep_vmap


def main():
    task = QuadCircle()
    # pol = task.nom_pol_vf
    pol = task.nom_pol_handcbf

    tf = 8.0
    result_dt = task.dt
    T = int(tf / result_dt)
    tf = T * result_dt
    print("T: {}".format(T))

    x0 = task.nominal_val_state()
    bb_x0, _, _ = task.get_ci_x0(0, 30)

    sim = SimCtsPbar(task, pol, T, result_dt)
    # T_states, T_ts, stats = jax2np(jax_jit(sim.rollout_plot)(x0))
    bbT_states, bbT_ts = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x0))

    # print(T_ts)
    ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        with jax.log_compiles():
            main()
