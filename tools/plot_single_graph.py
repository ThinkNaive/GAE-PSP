import sys

sys.path.append(sys.path[0].split("/")[0])

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_graph, count_accuracy, refine_distribution, post_process


if __name__ == "__main__":
    directory = "__runs_raw__/erdos-renyi3_gauss/d20_l1_1.0/_sd0_sm1"
    w_true_path = directory + "/graph_true.npy"
    w_est_path = directory + "/graph_estimated.npy"

    w_true = np.load(w_true_path)
    w_est = np.load(w_est_path)
    w_thr = post_process(refine_distribution(w_est))

    fig = plot_graph(
        [w_true, w_est],
        ["True Graph", "Estimated Graph"],
        ratio=1.0
    )
    plt.show()
