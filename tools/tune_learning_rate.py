import sys

sys.path.append(sys.path[0].split("/")[0])

import os
import random
import numpy as np
from multiprocessing import Pool
from utils import count_accuracy, post_process, denoise
from hyperparameter_tuning.tune_graph_threshold import refine_distribution


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


def _get_metric(tup):
    path, metric, threshold, lr = tup
    w_true = np.load(path[0])
    w_est = np.load(path[1])
    w_thr = post_process(refine_distribution(w_est), threshold)
    # w_thr = post_process(w_est, threshold)
    fdr, tpr, fpr, shd, nnz, gsc = count_accuracy(w_true, w_thr)
    metric = eval(metric)

    return lr, metric


def _get_average(results):
    res = {}

    for lr, metric in results:
        if lr not in res:
            res[lr] = [metric]
        else:
            res[lr].append(metric)
    
    for lr in res:
        res[lr] = np.mean(res[lr])

    return res


def list_metric(path, threshold, metric):
    args = []

    for key in path:
        lr = key.split("lr")[-1]

        argv = (
            path[key],
            metric,
            threshold,
            lr,
        )
        args.append(argv)

    random.shuffle(args)

    pool = Pool()
    ret = pool.map(_get_metric, args)

    res = _get_average(ret)

    print()
    return res


if __name__ == "__main__":
    # Arguments need to be specified
    # directory = "runs_for_sparsity/adam_wi_bn_lr_cut_sparsity"
    # directory = "runs_for_sparsity/adam_wi_bn_lr_log_sparsity"
    # directory = "runs_for_sparsity/adam_wo_bn_hard"
    # directory = "runs_gaussian/runs_adam_wo_bn"
    # directory = "runs_for_l1/gaussian_adam_wo_bn_ppt_l1"
    directory = "runs_for_l1/gaussian_adam_wo_bn_raw_l1"
    threshold = 0.08

    path = get_data_path(listdir(directory, filters=[""]))
    print(">>> Tune learning rate @ {} <<<".format(directory))
    print("Total number of tasks: {}".format(len(path)))
    print("=" * 40, "Running", "=" * 40)

    results = list_metric(path, threshold=threshold, metric="shd")

    best = list(results.keys())[0]
    for lr, val in sorted(list(results.items()), key=lambda t: t[0]):
        best = lr if results[lr] < results[best] else best
        print("lr={}  shd={:.1f}".format(lr, val))
    print("-" * 50, "Best metric", "-" * 50)
    print("lr={}  shd={:.1f}".format(best, results[best]))
