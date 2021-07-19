import os
import torch
from logger import init_logger
from utils import create_dir, get_output_dir, save_yaml_config, set_seed
from args import get_args
from dataset import SyntheticDataset
from model import CAE
from trainer import Trainer
from loguru import logger


def experiment():
    # get arguments
    args = get_args()

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.cuda) if args.cuda != -1 else "0"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    # setup log environment
    output_dir = get_output_dir(args)
    create_dir(output_dir)
    init_logger(output_dir, args.log_level)

    # save configs
    save_yaml_config(vars(args), path="{}/config.yaml".format(output_dir))
    logger.info("Run experiment - {}".format(output_dir))

    # generate dataset
    dataset = SyntheticDataset(
        args.n,
        args.d,
        args.graph_type,
        args.degree,
        args.sem_type,
        args.dataset_type,
        args.x_dim,
        args.seed_data,
    )
    logger.info("Generate {} series dataset with {} samples.".format(
        args.d, args.n))

    # fix model seed
    set_seed(args.seed_model)

    # create model
    model = CAE(args.d, args.x_dim, args.hidden_size, args.latent_dim,
                args.layer, args.lambda_sparsity, args.psp).double()
    model.to(device)
    logger.info("Create model with {} hidden size.".format(args.hidden_size))
    logger.debug(model)

    # instantiate trainer and record while training
    exp = Trainer(
        args.learning_rate,
        args.min_iters,
        args.init_rho,
        args.rho_thres,
        args.beta,
        args.gamma,
        args.min_h,
        args.max_h,
        args.early_stopping,
        args.mse_thres,
        device,
        output_dir,
    )
    logger.info("Start training with {} iterations.".format(args.max_iters))
    exp.train(
        model,
        dataset.X,
        dataset.W,
        args.max_iters,
        args.epochs,
        args.graph_thres,
    )


if __name__ == "__main__":
    experiment()
