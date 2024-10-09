import pathlib
import pickle

import jax.numpy as jnp
import scipy
from loguru import logger

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.utils.compare_ci import CIData
from clfrl.utils.jax_utils import jax2np, rep_vmap
from clfrl.utils.paths import get_data_dir, get_paper_data_dir


def main():
    data_dir = get_data_dir()
    mat_path = data_dir / "doubleint.mat"
    pkl_data_dir = get_paper_data_dir() / "dbint"

    mat = scipy.io.loadmat(str(mat_path))
    h_coefs = mat["h_coefs_double"]

    def eval_B(x):
        x1, x2 = x
        x1_terms = jnp.array([x1**2, x1, 1])
        x2_terms = jnp.array([x2**2, x2, 1])
        h_terms = x1_terms[:, None] * x2_terms[None, :]
        assert h_terms.shape == h_coefs.shape
        return jnp.sum(h_terms * h_coefs)

    task = DoubleIntWall()
    setup_idx = 0
    bb_x, bb_Xs, bb_Ys = task.get_paper_ci_x0(n_pts=128)
    bb_B = jax2np(rep_vmap(eval_B, rep=2)(bb_x))

    bbTh_h = -bb_B[:, :, None, None]

    ci_data = CIData("IntAvoid", task.name, setup_idx, None, None, bbTh_h, bb_Xs, bb_Ys)
    pkl_path = pkl_data_dir / f"sos.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ci_data, f)

    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    main()
