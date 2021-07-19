import sys

sys.path.append(sys.path[0].split("/")[0])

import os
import random
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt


def get_pure_noise_column(w_true, w_est):
    mask = np.sum(np.abs(w_true), axis=0) == 0
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
    sem, graph_set, res_bound = tup
    w_true = np.load(graph_set[0])
    w_est = np.load(graph_set[1])
    noise = get_pure_noise_column(w_true, w_est)
    # m = np.max(np.abs(w_est))
    # noise /= m

    sem.acquire()
    res_bound[0].append(np.min(noise))
    res_bound[1].append(np.max(noise))
    print(".", end="", flush=True)
    sem.release()


def _get_seeds(data):
    seeds = set()
    for key in data:
        seed = key.split("sd")[-1].split("_")[0]
        seeds.add(seed)
    return seeds


def _get_mean_level(results):
    min_noise = []
    max_noise = []

    for val in results.values():
        minim = np.mean(list(val[0]))
        maxim = np.mean(list(val[1]))
        min_noise.append(minim)
        max_noise.append(maxim)

    # min_noise = np.mean(min_noise)
    # max_noise = np.mean(max_noise)
    return min_noise, max_noise


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
    res_bound = {seed: (mg.list(), mg.list()) for seed in seeds}
    args = []

    for key in data:
        seed = key.split("sd")[-1].split("_")[0]

        argv = (sem[seed], data[key], res_bound[seed])
        args.append(argv)

    random.shuffle(args)

    pool = Pool()
    pool.map(_get_metric, args)

    pool.close()
    pool.join()

    min_noises, max_noises = _get_mean_level(res_bound)

    print()
    return min_noises, max_noises


if __name__ == "__main__":
    # Arguments need to be specified
    # directory = "runs_for_l1/gauss_raw_l1_1.0"
    directory = "runs_for_progressing"
    filters = ["l1_0.01"] #  ["lr0.0003", "sd0"]

    data = load_data(listdir(directory, filters=filters))
    print(">>> Show pure noise bound @ {} <<<".format(directory))
    print("Total number of tasks: {}".format(len(data)))
    print("=" * 40, "Running", "=" * 40)

    min_noises, max_noises = list_metric(data)

    for min, max in zip(min_noises, max_noises):
        print("min noise:{:.6f} max noise:{:.6f}".format(min, max))
    print("mean min noise:{:.6f} mean max noise:{:.6f}".format(np.median(min_noises), np.median(max_noises)))
