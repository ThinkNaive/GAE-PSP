import sys

sys.path.append(sys.path[0].split("/")[0])

from pathlib import Path
import random
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt


def listdir(base: str, pattern: str = "*"):
    files = Path(base).glob(pattern)
    res = sorted([file for file in files])
    return res


def _compute_weight_level(argv):
    idx, w_true_path, w_est_path = argv

    w_true = np.load(w_true_path)
    w_est = np.load(w_est_path)

    pos_median = np.median(np.abs(w_est[w_true != 0]))
    neg_median = np.median(np.abs(w_est[w_true == 0]))

    return idx, pos_median, neg_median


def _resolve_data(ret):
    x, pos, neg = [], [], []
    tmp = sorted([r for r in ret], key=lambda x: x[0])

    for idx, pos_median, neg_median in tmp:
        x.append(idx)
        pos.append(pos_median)
        neg.append(neg_median)

    return np.array(x), np.array(pos), np.array(neg)


def get_weight_level(directory):
    w_true_pattern = "graph_true.npy"
    w_ests_pattern = "w_est*"

    w_true = listdir(directory, w_true_pattern)
    assert len(w_true) == 1
    w_true = w_true[0]

    w_ests = listdir(directory, w_ests_pattern)

    args = [
        (int(str(w_est).split("w_est_")[-1].split(".npy")[0]), w_true, w_est)
        for w_est in w_ests
    ]

    pool = Pool()
    ret = pool.map(_compute_weight_level, args)

    x, pos, neg = _resolve_data(ret)

    return x, pos, neg


if __name__ == "__main__":
    # directory = "runs_for_progressing/d50_sd8_sm3_l1_1.0"
    directory = "runs_for_progressing/d50_sd8_sm3_l1_0.01"

    x, pos, neg = get_weight_level(directory)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_yscale('log')
    ax1.plot(x, pos)
    ax1.plot(x, neg)

    ax2 = fig.add_subplot(1, 2, 2)
    # ax2.set_yscale('log')
    ax2.plot(x, pos / neg)

    plt.show()
