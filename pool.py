import time
import os
import multiprocessing
import copy
import itertools

start_time = time.time()


def func(c, gpu):
    # Only when GPU is available, allocate it to command
    cmd = "{} --cuda={}".format(c, gpu)
    return os.system(cmd)
    # print(cmd)


def worker(task_sem, gpu_sem, gpus, command):

    # Safely allocating gpu
    task_sem.acquire()
    gpu_sem.acquire()
    gpu = gpus.pop()
    gpu_sem.release()

    # run command
    func(command, gpu)

    # Safely recycling gpu
    gpu_sem.acquire()
    gpus.append(gpu)
    gpu_sem.release()
    task_sem.release()


def mp(gpus, commands):
    """
    Multi-processing tasks with dynamic GPU allocation
    """
    with multiprocessing.Manager() as mg:
        gpus = mg.list(gpus)

        task_sem = mg.Semaphore(len(gpus))
        gpu_sem = mg.Semaphore(1)

        res = []

        for command in commands:
            p = multiprocessing.Process(
                target=worker, args=(task_sem, gpu_sem, gpus, command)
            )
            res.append(p)
            p.start()
        for p in res:
            p.join()


def generate_commands(params):
    cmd = "python main.py"

    for key in list(params):
        if not isinstance(params[key], list):
            val = params.pop(key)
            if isinstance(val, bool):
                if val:
                    cmd += " {}".format(key)
            else:
                cmd += " {}={}".format(key, val)

    cmds = []
    tags = list(params)
    comb = (v for v in params.values())

    for res in itertools.product(*comb):
        c = copy.copy(cmd)

        for k, v in zip(tags, res):
            c += " {}={}".format(k, v)

        cmds.append(c)

    return cmds


if __name__ == "__main__":
    # Available GPU list
    gpus: list = [0, 1, 2, 3, 4, 5, 6, 7]
    # Number of tasks running simultaneously on a GPU
    n_task_for_each: int = 3
    # Parameter range
    params = {}
    params["--n"] = 3000  # 3000, 1000
    params["--d"] = [10, 20, 50, 100]  # [10, 20, 50, 100]
    params["--graph_type"] = "erdos-renyi"  # ["erdos-renyi", "barabasi-albert"], ["ER", "SF", "BP"]
    params["--degree"] = 3  # 3, [2, 4]
    params["--sem_type"] = "gauss"  # ['gauss', 'exp', 'mnonr'], ["mlp", "mim", "gp", "gp-add"]
    params["--dataset_type"] = "nonlinear_3"  # ["nonlinear_1", "nonlinear_2", "nonlinear_3"], ["Undefined"]
    params["--x_dim"] = 1
    params["--hidden_size"] = 16
    params["--layer"] = 3
    params["--latent_dim"] = 1
    params["--lambda_sparsity"] = 0.01  # [0.001, 0.01, 0.1, 1.0]
    params["--psp"] = True  # Whether to use proportional sparsity penalty (True, False)
    params["--learning_rate"] = 3e-4  # note: 3e-4 is the best
    params["--max_iters"] = 20
    params["--min_iters"] = 5
    params["--epochs"] = 300
    params["--init_rho"] = 1.0
    params["--rho_thres"] = 1e18
    params["--beta"] = 10.0
    params["--gamma"] = 0.25
    params["--min_h"] = 1e-12
    params["--max_h"] = 1e-7
    params["--early_stopping"] = True
    params["--mse_thres"] = 1.15
    params["--seed_data"] = [i for i in range(10)]
    params["--seed_model"] = [i for i in range(10)]
    params["--graph_thres"] = 0.20  # 0.08, 0.10, 0.20
    params["--base_dir"] = "__runs_psp__"  # psp raw
    params["--log_level"] = "DEBUG"

    # run
    commands = generate_commands(params)
    print("Total number of tasks: {}".format(len(commands)))
    print("=" * 40, "Running", "=" * 40)
    mp(gpus * n_task_for_each, commands)
