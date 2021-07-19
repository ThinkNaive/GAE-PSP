import copy
import sys

sys.path.append(sys.path[0].split("/")[0])

import os
import random
import numpy as np
from multiprocessing import Pool
from utils import count_accuracy, post_process
from loguru import logger


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
    path = {}
    for dir in dirs:
        key = os.path.split(dir)[-1]
        path[key] = _load_graph_path(dir)
    return path


def _get_metric(tup):
    path, metrics, threshold = tup
    w_true = np.load(path[0])
    w_est = np.load(path[1])
    w_thr = post_process(w_est, threshold)
    # w_thr = post_process(refine_distribution(w_est), threshold)
    fdr, tpr, fpr, shd, nnz, gsc = count_accuracy(w_true, w_thr)
    acc = {
        "fdr": fdr,
        "tpr": tpr,
        "fpr": fpr,
        "shd": shd,
        "nnz": nnz,
        "gsc": gsc
    }
    res = {metric: acc[metric] for metric in metrics}
    # print(threshold, metric)
    return threshold, res


def _get_average(results):
    res = {}

    for threshold, metrics in results:
        if threshold not in res:
            res[threshold] = {key: [metrics[key]] for key in metrics}
        else:
            for key in metrics:
                res[threshold][key].append(metrics[key])

    for threshold in res:
        for key in res[threshold]:
            res[threshold][key] = np.mean(res[threshold][key]), np.std(res[threshold][key], ddof=1)

    return res


def list_metrics(path, thresholds, metrics):
    args = []

    for key in path:
        for threshold in thresholds:
            argv = (
                path[key],
                metrics,
                threshold,
            )
            args.append(argv)

    random.shuffle(args)

    pool = Pool()
    ret = pool.map(_get_metric, args)

    res = _get_average(ret)

    return res


def store_tune_results(res, filename):
    n = len(res.keys())
    m = np.zeros((n, 2))
    for i, (thr, val) in enumerate(sorted(res.items())):
        m[i, 0] = thr
        m[i, 1] = val
    np.save(filename, m)


if __name__ == "__main__":
    # Arguments need to be specified
    directory = "__runs_psp__/erdos-renyi3_gauss/d100_l1_0.01"
    # directory = "__runs_raw__/erdos-renyi3_gauss/d100_l1_1.0"
    filters = [""]
    # filters = ["d20_", "l1_0.01"]  # ["d10_", "l1_0.001"]  [""]
    # filter = "lr0.0003"
    expect_thr = 0.2
    metrics = ["shd", "tpr", "gsc"]

    # prog
    path = get_data_path(listdir(directory, filters=filters))
    thresholds = [i for i in [k * 0.01 for k in range(1, 51)]]
    print(">>> Tune graph threshold by {} @ {} <<<".format(filters, directory))
    print("Total number of tasks: {}".format(len(thresholds) * len(path)))
    print("=" * 40, "Running", "=" * 40)

    results = list_metrics(path, thresholds, metrics)

    best = list(results.keys())[0]
    for thr, vals in sorted(results.items()):
        val, std = vals[metrics[0]]
        best = thr if val < results[best][metrics[0]][0] else best

        print("threshold={:.2f}".format(thr), end="")
        [print("  {}:{:.2f}±{:.2f}".format(key, ave, std), end="") for key, (ave, std) in vals.items()]
        print()

    print("-" * 50, "Best metric", "-" * 50)
    print("threshold={:.2f}".format(best), end="")
    [print("  {}:{:.2f}±{:.2f}".format(key, ave, std), end="") for key, (ave, std) in results[best].items()]
    print()
    # print expected threshold result
    print("-" * 50, "Threshold {}".format(expect_thr), "-" * 50)
    print("threshold={:.2f}".format(expect_thr), end="")
    [print("  {}:{:.2f}±{:.2f}".format(key, ave, std), end="") for key, (ave, std) in results[expect_thr].items()]
    print()
    # store_tune_results(results, "result_tune_threshold.npy")
