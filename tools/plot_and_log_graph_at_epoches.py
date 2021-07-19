import sys

sys.path.append(sys.path[0].split("/")[0])

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_graph, count_accuracy, refine_distribution, post_process


def listdir(base: str, pattern: str = "*"):
    files = Path(base).glob(pattern)
    res = sorted([file for file in files])
    return res


if __name__ == "__main__":
    directory = "runs_for_progressing/d50_sd8_sm3_l1_1.0"
    w_true_pattern = "graph_true.npy"
    w_est_patterns = "w_est*"

    w_true = np.load(listdir(directory, w_true_pattern)[0])
    w_ests_path = listdir(directory, w_est_patterns)

    for w_est_path in w_ests_path:
        index = Path(w_est_path).parts[-1].split(".")[0].split("w_est_")[-1]
        w_est = np.load(w_est_path)

        w_thr = post_process(refine_distribution(w_est), graph_thres=0.2)
        metrics = count_accuracy(w_true, w_thr)
        print(
            "Index:{:04d} FDR:{:7.2%} TPR:{:7.2%} FPR:{:7.2%} SHD:{:3d} NNZ:{:3d} GSC:{:7.2%}".format(
                int(index), *metrics
            )
        )

        fig = plot_graph(
            [w_true, w_est, w_thr],
            ["True Graph", "Estimated Graph", "Thresholded Graph"],
            "Iteration - {}".format(int(index)),
            ratio=10.0,
            keep_absolute=True,
        )
        fig.savefig(
            Path(directory).joinpath("iter_{:04d}.png".format(int(index))),
            transparent=True,
        )
