import os
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import pathlib
import copy


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except:
        pass


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.

    Code from:
        https://github.com/ignavier/golem/blob/main/src/utils/utils.py
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def count_accuracy(W_true, W_est) -> tuple:
    """Compute FDR, TPR, and FPR for B

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
        gsc: g-score

    Code modified from:
        https://github.com/fishmoon1234/DAG-GNN/blob/master/src/utils.py
    """
    B_true = W_true != 0
    B_est = W_est != 0
    d = min(B_est.shape)

    # fdr
    fp = np.sum(~B_true & B_est)
    pp = np.sum(B_est)
    fdr = fp / pp

    # tpr
    tp = np.sum(B_true & B_est)
    tt = np.sum(B_true)
    tpr = tp / tt

    # fpr
    tf = d * (d - 1) / 2 - np.sum(B_true) + B_est.shape[0] * B_est.shape[1] - d * d
    fpr = fp / tf

    # shd
    shd = np.sum(B_true != B_est)

    # nnz
    nnz = pp

    # g-score
    fn = np.sum(B_true & ~B_est)
    gsc = max(0, tp - fp) / (tp + fn)

    return fdr, tpr, fpr, shd, nnz, gsc


def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.

    Code from:
        https://github.com/ignavier/golem/blob/main/src/utils/utils.py
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(
        zip(B[nonzero_indices], nonzero_indices[0], nonzero_indices[1])
    )
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def denoise(w):
    """Denoise estimated weighted matrix:
    There is a phenomenon that each column of the estimated matrix
    prones to have bias (either positive or negative). Inspired by
    it, together with the sparsity assumption, we remove bias by
    substract mean value of the minor elements in each column.
    """
    # ord = np.argsort(np.abs(w), axis=0)
    # rep = np.zeros(w.shape)

    # for i in range(len(ord)):
    #     for j in range(len(ord[i])):
    #         rep[i, j] = w[ord[i][j], j]

    # t = int(w.shape[0] * 0.75)
    # drift = np.mean(rep[:t, :], axis=0)
    # mask = (w > drift) * (drift > 0) + (w < drift) * (drift < 0)
    # w_star = (w - drift) * mask + w * (1 - mask)

    rep = np.sort(w, axis=0)
    s = int(w.shape[0] * 0.2)
    t = int(w.shape[0] * 0.8)
    drift = np.mean(rep[s:t, :], axis=0)
    return w - drift


def refine_distribution(w, noise_level_percentage=0.05):
    w = copy.deepcopy(w)
    pos_cut = noise_level_percentage * np.max(w)
    neg_cut = noise_level_percentage * np.min(w)

    w_pos = w[w > pos_cut]
    if w_pos.size:
        q1 = np.percentile(w_pos, 25, interpolation="lower")
        q3 = np.percentile(w_pos, 75, interpolation="higher")
        iqr = q3 - q1
        ub = q3 + 1.5 * iqr
        w[w > ub] = np.max(w_pos[w_pos <= ub])

    w_neg = w[w < neg_cut]
    if w_neg.size:
        q1 = np.percentile(w_neg, 25, interpolation="lower")
        q3 = np.percentile(w_neg, 75, interpolation="higher")
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        w[w < lb] = np.min(w_neg[w_neg >= lb])

    return w


def post_process(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained. (deprecated)

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.

    Code modified from:
        https://github.com/ignavier/golem/blob/main/src/utils/utils.py

        https://github.com/huawei-noah/trustworthyAI/blob/master/Causal_Structure_Learning/GAE_Causal_Structure_Learning/src/trainers/al_trainer.py
    """
    B = B / np.max(np.abs(B))
    B[np.abs(B) < graph_thres] = 0  # Thresholding
    # B, _ = threshold_till_dag(B)

    return B


def plot_graph(
    W_set: list,
    T_set: list,
    title: str = "",
    ratio=2.0,
    keep_absolute=False,
    shape=None,
    figratio=2.5,
):
    """A tool for ploting heat maps

    Args:
        W_set (list): A list of 2 dimensional matrixes with np.ndarray format.add()
        T_set (list): A list of titles with str format.
        title (str, optional): Main title. Defaults to "".
        ratio (float, optional): A parameter for controlling color depth. Defaults to 2.0.
        keep_absolute (bool, optional): Whether normalize matrix weight to range (-1, 1). Defaults to False.
        shape (tuple(int, int)), optional): Specify graph arrangement. If None, use one row and len(W_set) columns. Defaults to None.
        figratio (int, optional): A parameter for controlling figure size. Defaults to 4.

    Returns:
        Figure: figure handler
    """
    assert shape == None or len(shape) == 2

    if not isinstance(W_set, list):
        W_set = [W_set]
    if not isinstance(T_set, list):
        T_set = [T_set]

    size = len(W_set)

    def _init():
        plot_graph.fig = (
            plt.figure(figsize=(figratio * size, figratio))
            if shape is None
            else plt.figure(figsize=(figratio * shape[1], figratio * shape[0]))
        )
        plot_graph.fig.set_tight_layout(True)
        plot_graph.axes = [
            plot_graph.fig.add_subplot(1, size, i)
            if shape is None
            else plot_graph.fig.add_subplot(shape[0], shape[1], i)
            for i in range(1, size + 1)
        ]

    # initialize subplots once for all
    if not hasattr(plot_graph, "axes"):
        _init()
    elif len(plot_graph.axes) != size:
        _init()
    else:
        for ax in plot_graph.axes:
            ax.cla()

    fig = plot_graph.fig
    axes = plot_graph.axes
    fig.suptitle(title)

    # Plot ground truth
    for i, (W, T) in enumerate(zip(W_set, T_set)):
        if not keep_absolute:
            W = W / np.max(np.abs(W)) * ratio
        else:
            W *= ratio
        axes[i].imshow(W, cmap="seismic", interpolation="none", vmin=-1.0, vmax=1.0)
        axes[i].set_title(T)

    return fig


def plot_solution(B_true, B_est, graph_threshold=0.3):
    """Checkpointing after the training ends.

    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_proc (numpy.ndarray): [d, d] post-processed weighted matrix.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.

    Code modified from:
        https://github.com/ignavier/golem/blob/main/src/utils/utils.py
    """
    fig, axes = plt.subplots(figsize=(10, 3), ncols=3)

    # Plot ground truth
    B_true = B_true / np.max(np.abs(B_true)) * 2.0
    im = axes[0].imshow(
        B_true, cmap="seismic", interpolation="none", vmin=-1.0, vmax=1.0
    )
    axes[0].set_title("Ground truth", fontsize=13)
    axes[0].tick_params(labelsize=13)

    # Plot estimated solution
    B_est = B_est / np.max(np.abs(B_est)) * 2.0
    im = axes[1].imshow(
        B_est, cmap="seismic", interpolation="none", vmin=-1.0, vmax=1.0
    )
    axes[1].set_title("Estimated graph", fontsize=13)
    axes[1].set_yticklabels([])  # Remove yticks
    axes[1].tick_params(labelsize=13)

    # Plot post-processed solution
    B_proc = post_process(B_est, graph_threshold) * 2.0
    im = axes[2].imshow(
        B_proc, cmap="seismic", interpolation="none", vmin=-1.0, vmax=1.0
    )
    axes[2].set_title("Threshold {:.1f} graph".format(graph_threshold), fontsize=13)
    axes[2].set_yticklabels([])  # Remove yticks
    axes[2].tick_params(labelsize=13)

    # Adjust space between subplots
    fig.subplots_adjust(wspace=0.1)

    # Colorbar (with abit of hard-coding)
    im_ratio = 3 / 10
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), fraction=0.05 * im_ratio, pad=0.035
    )
    cbar.ax.tick_params(labelsize=13)

    return fig


def get_output_dir(args):
    base = args.base_dir

    tag_gt = str(args.graph_type)
    tag_dg = str(args.degree)
    tag_st = "_{}".format(args.sem_type)

    tag_d = "d{}".format(args.d)
    tag_l1 = "_l1_{}".format(args.lambda_sparsity)

    tag_sd = "_sd{}".format(args.seed_data)
    tag_sm = "_sm{}".format(args.seed_model)

    spec = tag_gt + tag_dg + tag_st + "/" + tag_d + tag_l1 + "/" + tag_sd + tag_sm
    return "{}/{}".format(base, spec)


def load_yaml_config(path):
    """Load the config file in yaml format.

    Args:
        path (str): Path to load the config file.

    Returns:
        dict: config.

    Code from:
        https://github.com/ignavier/golem/blob/main/src/utils/config.py
    """
    with open(path, "r") as infile:
        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    """Load the config file in yaml format.

    Args:
        config (dict object): Config.
        path (str): Path to save the config.

    Code from:
        https://github.com/ignavier/golem/blob/main/src/utils/config.py
    """
    with open(path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def create_dir(output_dir):
    """Create directory.

    Args:
        output_dir (str): A directory to create if not found.

    Returns:
        exit_code: 0 (success) or -1 (failed).

    Code from:
        https://github.com/ignavier/golem/blob/main/src/utils/dir.py
    """
    try:
        if not pathlib.Path(output_dir).exists():
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as err:
        # _logger.critical("Error when creating directory: {}.".format(err))
        exit(-1)
