Source code for "Incorporating Proportional Sparse Penalty for Causal Structure Learning"

And the

Reimplementation of "A Graph Autoencoder Approach to Causal Structure Learning" (https://arxiv.org/abs/1911.07420)

---
# Installation
## Test Environment
    CUDA Version 9.0.176
    CUDNN Version 7.6.5
    Python 3.6.13

## Requirements
    loguru==0.5.3
    matplotlib==3.3.4
    networkx==2.5.1
    numpy
    scipy
    pandas==1.1.5
    Pillow==8.2.0
    PyYAML==5.4.1
    tensorboard==2.5.0
    torch==1.9.0
    yapf==0.31.0

---
# Run
## Single experiment

- Run GAE-PSP

Run with

    python main.py --n=3000 --d=100 --graph_type=erdos-renyi --degree=3 --sem_type=gauss --dataset_type=nonlinear_3 --x_dim=1 --hidden_size=16 --latent_dim=1 --lambda_sparsity=0.01 --layer=3 --learning_rate=1e-3 --max_iters=20 --min_iters=5 --epochs=300 --init_rho=1.0 --rho_thres=1e18 --beta=10.0 --gamma=0.25 --min_h=1e-12 --max_h=1e-7 --mse_thres=1.15 --seed_data=0 --seed_model=0 --graph_thres=0.2 --base_dir=runs --cuda=-1 --psp --log_level=INFO

- Run GAE

Run with

    python main.py --n=3000 --d=100 --graph_type=erdos-renyi --degree=3 --sem_type=gauss --dataset_type=nonlinear_3 --x_dim=1 --hidden_size=16 --latent_dim=1 --lambda_sparsity=1.0 --layer=3 --learning_rate=1e-3 --max_iters=20 --min_iters=5 --epochs=300 --init_rho=1.0 --rho_thres=1e18 --beta=10.0 --gamma=0.25 --min_h=1e-12 --max_h=1e-7 --mse_thres=1.15 --seed_data=0 --seed_model=0 --graph_thres=0.2 --base_dir=runs --cuda=-1 --log_level=INFO

## Multiple experiments
This task will run multiple experiments with varying parameters (seed for dataset and training) distributed dynamically to the pre-defined GPU list.

Firstly, set parameters `gpus`, `seed_data_range` and `seed_model_range` in `pool.py`. Here is an example:

    # Available GPU list
    gpus: list = [0, 1, 2, 3, 4, 5, 6, 7]  # up to number of devices, add -1 means add CPU
    # Number of tasks running simultaneously on a GPU/CPU for fully utilizing computing power
    n_task_for_each: int = 3
    # Parameter arrangement, scalar value means only one experiment, while list of values means multiple experiment for different values 
    params = {}
    params["--n"] = 3000  # number of samples
    params["--d"] = [10, 20, 50, 100]  # experiment with different number of variables
    params["--graph_type"] = "erdos-renyi"
    params["--degree"] = 3
    params["--sem_type"] = "gauss"  # ["gauss", "mnonr"]
    params["--dataset_type"] = "nonlinear_3"
    params["--x_dim"] = 1
    params["--hidden_size"] = 16
    params["--layer"] = 3
    params["--latent_dim"] = 1
    params["--lambda_sparsity"] = 0.01  # l1-regulartion (0.01, 1.0)
    params["--psp"] = True  # (True, False)
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
    params["--seed_data"] = [i for i in range(10)]  # experiments for 10 dataset with different seeds
    params["--seed_model"] = [i for i in range(10)]  # experiments for 10 dataset with different model initializations
    params["--graph_thres"] = 0.20  # default 0.2
    params["--base_dir"] = "runs"  # which to store results
    params["--log_level"] = "DEBUG"  # if print and store intermediate outputs

Then, run command as follows:

    python pool.py

# Evaluate
Simply use thresholding tool for evaluating results, after specifying `directory` in `tune_graph_threshold.py`, run this:

    cd tools
    python tune_graph_threshold.py
