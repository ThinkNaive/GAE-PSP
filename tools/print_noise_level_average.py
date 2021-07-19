import sys

sys.path.append(sys.path[0].split("/")[0])

import os
import random
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt

n_digit = 3


def _category(matrix, res, n):
    for row in matrix:
        for e in row:
            key = round(e - 0.5*10**(-n_digit), ndigits=3)
            if key not in res:
                res[key] = 1
            else:
                res[key] += 1
    
    for key in res:
        res[key] /= n


def _get_hist(w_true, w_est):
    noise = (w_true == 0) * w_est
    pred = w_est - noise

    r_pred = {}
    r_noise = {}

    n_pred = np.sum(w_true != 0)
    n_noise = np.sum(w_true == 0)

    _category(pred, r_pred, n_pred)
    _category(noise, r_noise, n_noise)

    return r_pred, r_noise


def get_noise(w_true, w_est):
    mask = w_true == 0
    noise = w_est * mask
    noise = np.sort(noise, axis=0)
    return noise


def _store_hist(dist, source):
    for key in source:
        if key not in dist:
            dist[key] = source[key]
        else:
            dist[key] += source[key]


def _get_metric(tup):
    sem, graph_set, res_mean, res_hist = tup
    graph_true = np.load(graph_set[0])
    graph_est = np.load(graph_set[1])
    noise = get_noise(graph_true, graph_est)
    m = np.max(np.abs(noise))
    noise /= m

    l_true = np.sum(np.abs(graph_true), axis=0)
    l_true /= np.max(l_true)
    l_est = np.sum(np.abs(graph_est), axis=0)
    l_est /= np.max(l_est)
    l_noise = np.sum(np.abs(noise), axis=0)
    l_noise /= np.max(l_noise)

    mean_estn_positive = np.mean(l_est[l_true != 0])
    mean_estn_negative = np.mean(l_est[l_true == 0])

    r_pred, r_noise = _get_hist(graph_true, graph_est)

    sem.acquire()
    res_mean[0].append(mean_estn_positive)
    res_mean[1].append(mean_estn_negative)
    _store_hist(res_hist["pred"], r_pred)
    _store_hist(res_hist["noise"], r_noise)
    print(".", end="", flush=True)
    sem.release()


def _get_seeds(data):
    seeds = set()
    for key in data:
        seed = key.split("sd")[-1].split("_")[0]
        seeds.add(seed)
    return seeds


def _get_mean_level(results):
    positive = []
    negative = []

    for val in results.values():
        pos = np.mean(list(val[0]))
        neg = np.mean(list(val[1]))
        positive.append(pos)
        negative.append(neg)

    positive = np.mean(positive)
    negative = np.mean(negative)
    return positive, negative


def _get_range(results):
    min_x = float("inf")
    max_x = float("-inf")

    for result in results.values():
        for key in result["pred"]:
            if min_x > key:
                min_x = key
            if max_x < key:
                max_x = key

        for key in result["noise"]:
            if min_x > key:
                min_x = key
            if max_x < key:
                max_x = key

    return np.arange(min_x, max_x + 10**(-n_digit), 10**(-n_digit))


def _store_histgram(y_list, y_dict, x_range):
    for key in y_dict:
        pos = np.argwhere((x_range > key - 0.5*10**(-n_digit)) * (x_range < key + 0.5*10**(-n_digit))).squeeze()
        if -0.5*10**(-n_digit) < key < 0.5*10**(-n_digit):
            continue
        y_list[int(pos)] = y_dict[key]


def _get_histgram(results):
    x_range = _get_range(results)
    y_pred = {key: [0 for _ in x_range] for key in results}
    y_noise = {key: [0 for _ in x_range] for key in results}

    for key in results:
        _store_histgram(y_pred[key], results[key]["pred"], x_range)
        _store_histgram(y_noise[key], results[key]["noise"], x_range)

    return x_range, y_pred, y_noise


def listdir(base: str, filters: str = ""):
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


def load_data(dirs):
    data = {}
    for dir in dirs:
        key = os.path.split(dir)[-1]
        data[key] = _load_graph_path(dir)
    return data


def list_metric(data):
    seeds = _get_seeds(data)
    mg = multiprocessing.Manager()
    sem = {seed: mg.Semaphore(1) for seed in seeds}
    res_mean = {seed: (mg.list(), mg.list()) for seed in seeds}
    res_hist = {
        seed: {"pred": mg.dict(), "noise": mg.dict()}
        for seed in seeds
    }
    args = []

    for key in data:
        seed = key.split("sd")[-1].split("_")[0]

        argv = (sem[seed], data[key], res_mean[seed], res_hist[seed])
        args.append(argv)

    random.shuffle(args)

    pool = Pool()
    pool.map(_get_metric, args)

    pool.close()
    pool.join()

    pos_mean, neg_mean = _get_mean_level(res_mean)
    x_range, y_pred, y_noise = _get_histgram(res_hist)

    print()
    return pos_mean, neg_mean, x_range, y_pred, y_noise


def plot_hist(x_range, y_pred, y_noise):
    params = [
        (y_pred, "true positive"),
        (y_noise, "negative noise"),
    ]

    fig, axes = plt.subplots(figsize=(3 * len(params), 3), ncols=len(params))

    for i, param in enumerate(params):
        axes[i].bar(x_range, param[0], width=0.2*10**(-n_digit))
        axes[i].set_yscale("log")
        axes[i].set_title("Distribution of {}".format(param[1]))

    return fig


if __name__ == "__main__":
    # Arguments need to be specified
    # directory = "runs_for_l1/gaussian_adam_wo_bn_psp_l1_1.0"
    directory = "runs_for_progressing"
    filters = ["l1_1.0"] #  ["lr0.0003", "sd0"]

    data = load_data(listdir(directory, filters=filters))
    print(">>> Show noise between postive and negative @ {} <<<".format(directory))
    print("Total number of tasks: {}".format(len(data)))
    print("=" * 40, "Running", "=" * 40)

    pos, neg, x_range, y_pred, y_noise = list_metric(data)

    print("mean_positive(true):{:.3f} mean_negative(noise):{:.3f}".format(pos, neg))

    # for key in y_pred:
    #     plot_hist(x_range, y_pred[key], y_noise[key])
    # plt.show()
