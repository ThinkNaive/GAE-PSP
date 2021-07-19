import sys

sys.path.append(sys.path[0].split("/")[0])

import copy
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from utils import post_process, denoise, count_accuracy, create_dir


def listdir(base: str, filters: str = [""]):
    res = []
    dirs = os.listdir(base)
    for d in dirs:
        d = os.path.join(base, d)
        if os.path.isdir(d):
            if all(filter in d for filter in filters):
                res.append(d)
    return res


def _load_graph_path(path):
    path_true = os.path.join(path, "graph_true.npy")
    path_esti = os.path.join(path, "graph_estimated.npy")
    return path_true, path_esti


def get_data_path(dirs):
    data = {}
    for dir in dirs:
        key = os.path.split(dir)[-1]
        data[key] = _load_graph_path(dir)
    return data


def refine_distribution(w, noise_level_percentage=0.05):
    w = copy.deepcopy(w)
    cut = noise_level_percentage * np.max(np.abs(w))

    w_pos = w[w > cut]
    if w_pos.size:
        q1 = np.percentile(w_pos, 25, interpolation="lower")
        q3 = np.percentile(w_pos, 75, interpolation="higher")
        iqr = q3 - q1
        ub = q3 + 1.5 * iqr
        w[w > ub] = np.max(w_pos[w_pos <= ub])

    w_neg = w[w < -cut]
    if w_neg.size:
        q1 = np.percentile(w_neg, 25, interpolation="lower")
        q3 = np.percentile(w_neg, 75, interpolation="higher")
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        w[w < lb] = np.min(w_neg[w_neg >= lb])

    return w


def plot_graph(W_set: list, T_set: list):
    if not isinstance(W_set, list):
        W_set = [W_set]
    if not isinstance(T_set, list):
        T_set = [T_set]

    size = len(W_set)

    # initialize subplots once for all
    if not hasattr(plot_graph, "fig"):
        plot_graph.fig = plt.figure(figsize=(8, 4))
    else:
        plot_graph.fig.clf()

    fig = plot_graph.fig
    axes = [fig.add_subplot(1, size, i) for i in range(1, size + 1)]

    # Plot ground truth
    for i, (W, T) in enumerate(zip(W_set, T_set)):
        W = W / np.max(np.abs(W)) * 2.0
        axes[i].imshow(W, cmap="seismic", interpolation="none", vmin=-1.0, vmax=1.0)
        axes[i].set_title(T)

    return fig


def print_metric(tup, title):
    fdr, tpr, fpr, shd, nnz = tup
    print(title, end=" ")
    print(
        "FDR:{:.2%} TPR:{:.2%} FPR:{:.2%} SHD:{:2d} NNZ:{:2d}".format(
            fdr, tpr, fpr, shd, nnz
        )
    )


def save_graph(path, titles, output, file):
    w_true = np.load(path[0])
    w_est = np.load(path[1])

    fig = plot_graph([w_true, w_est], titles)

    figname = file + ".png"
    fig.savefig(Path(output).joinpath(figname), dpi=300, transparent=True)


if __name__ == "__main__":
    base = "__output_images__"

    directory = "runs_for_l1/gauss_raw_l1_1.0"
    # directory = "runs_for_l1/gauss_psp_l1_1.0_d^2"
    filters = [""] # ["d100_"]

    output = Path(base).joinpath(directory.split('/')[-1])
    create_dir(output)
    path = get_data_path(listdir(directory, filters=filters))

    for file in sorted(list(path.keys())):
        save_graph(path[file], ["True Graph", "Estimated Graph"], output, file)
        print("Process {}".format(file))
