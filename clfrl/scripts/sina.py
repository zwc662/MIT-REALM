import ipdb
import matplotlib.pyplot as plt
import numpy as np

from clfrl.plotting.sinaplot import sinaplot


def main():
    rng = np.random.default_rng(seed=57812)
    batch_size = 1024
    data = rng.normal(0, 1, batch_size)

    # sinaplot(y=data)
    # plt.show()

    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.boxplot


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
