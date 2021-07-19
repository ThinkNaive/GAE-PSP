import sys
import argparse


def get_args():
    """
    Code modified from:
        https://github.com/ignavier/golem/blob/main/src/utils/config.py
    """
    parser = argparse.ArgumentParser()

    # configurations for dataset
    parser.add_argument(
        "--n",
        type=int,
        default=3000,
        help="Number of samples.",
    )

    parser.add_argument(
        "--d",
        type=int,
        default=20,
        help="Number of nodes.",
    )

    parser.add_argument(
        "--graph_type",
        type=str,
        default="erdos-renyi",
        help="Type of graph ('erdos-renyi', 'barabasi-albert').",
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Degree of graph.",
    )

    parser.add_argument(
        "--sem_type",
        type=str,
        default="gauss",
        help="Type of noise ['gauss', 'exp', 'gumbel', 'mnonr'].",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="nonlinear_3",
        help="Choose between nonlinear_1, nonlinear_2, nonlinear_3.",
    )

    parser.add_argument(
        "--x_dim",
        type=int,
        default=1,
        help="Dimension of vector for X.",
    )

    # configurations for model
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Hidden size for NonLinerTransformer.",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=3,
        help="Number of MLP layers for both encoder and decoder.",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1,
        help="Latent dimension for CAE.",
    )

    parser.add_argument(
        "--lambda_sparsity",
        type=float,
        default=1.0,
        help="Coefficient of L1 penalty.",
    )

    parser.add_argument(
        "--psp",
        action="store_true",
        help="Whether to use proportional sparsity penalty.",
    )

    # configurations for training
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="Upper bound of iterations for ALM optimization.",
    )

    parser.add_argument(
        "--min_iters",
        type=int,
        default=5,
        help="Lower bound of iterations for early stopping.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for training in each iterations.",
    )

    parser.add_argument(
        "--init_rho",
        type=float,
        default=1.0,
        help="Initial value for rho.",
    )

    parser.add_argument(
        "--rho_thres",
        type=float,
        default=1e18,
        help="Threshold for rho.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Multiplication to amplify rho each time.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="Threshold for judge h(A).",
    )

    parser.add_argument(
        "--min_h",
        type=float,
        default=1e-12,
        help="Lower bound of h for early stopping.",
    )

    parser.add_argument(
        "--max_h",
        type=float,
        default=1e-7,
        help="Upper bound of h for early stopping.",
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to use early stopping.",
    )

    parser.add_argument(
        "--mse_thres",
        type=float,
        default=1.15,
        help="Threshold ratio between reconstruction loss for early stopping",
    )

    # configurations for others
    parser.add_argument(
        "--seed_data",
        type=int,
        default=0,
        help="Random seed for generating dataset.",
    )

    parser.add_argument(
        "--seed_model",
        type=int,
        default=0,
        help="Random seed for initializing model.",
    )

    parser.add_argument(
        "--graph_thres",
        type=float,
        default=0.3,
        help="Threshold to filter out small values in graph",
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs",
        help="Base output folder.",
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="Whether to use GPU for training. (-1 for CPU, and 0,...,n for GPU)",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        help="log level (INFO, DEBUG).",
    )

    # parse arguments
    return parser.parse_args(args=sys.argv[1:])
