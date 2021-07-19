from pathlib import Path
import time
import numpy as np
from numpy.core.defchararray import mod
import torch
from torch.autograd.variable import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import count_accuracy, plot_graph, post_process, refine_distribution
from loguru import logger


class Trainer:
    def __init__(
        self,
        learning_rate,
        min_iters,
        init_rho,
        rho_thres,
        beta,
        gamma,
        min_h,
        max_h,
        early_stopping,
        mse_thres,
        device,
        output_dir,
    ):
        self._lr = learning_rate
        self._min_iters = min_iters
        self._init_rho = init_rho
        self._rho_thres = rho_thres
        self._beta = beta
        self._gamma = gamma
        self._min_h = min_h
        self._max_h = max_h
        self._early_stopping = early_stopping
        self._mse_thres = mse_thres
        self._device = device
        self._writer = SummaryWriter(output_dir)
        self._output_dir = output_dir

    def train(
        self,
        model,
        X,
        W_true,
        max_iters,
        epochs,
        graph_threshold,
    ):
        # for test only
        self._W_true = W_true
        self._thr = graph_threshold
        self._i = 0

        # record time consume
        start_time = time.time()

        # set optimizer
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        # optimizer = optim.RMSprop(model.parameters(), lr=self._lr)

        # prepare dataset
        np.save(Path(self._output_dir).joinpath("X.npy"), X)
        X = torch.FloatTensor(X)
        X = X.to(self._device)
        X = Variable(X, requires_grad=False).double()

        # set train mode
        model.train()

        # initialize temporary parameters
        alpha, rho = 0.0, self._init_rho
        h_prev, W_prev, mse_prev = np.inf, None, float("inf")

        for iter in range(1, max_iters + 1):
            # log iteration
            logger.debug("========== Iteration {} training ==========".format(iter))

            while rho < self._rho_thres:
                # train with certain alpha and rho
                loss, loss_mse, h_new, W_new = self.train_epochs(
                    model, X, optimizer, epochs, alpha, rho
                )

                # strategy for adjusting rho
                if h_new > self._gamma * h_prev:
                    rho *= self._beta
                else:
                    break

            # MSE increases too much, perform early stopping when h_est is sufficiently small
            if self._early_stopping:
                if loss_mse / mse_prev > self._mse_thres and h_new <= self._max_h:
                    logger.info("Early stopping at {}-th iteration".format(iter))
                    break
                else:
                    mse_prev = loss_mse

            # compute metrics
            W_thr = post_process(W_new, graph_threshold)
            fdr, tpr, fpr, shd, nnz, gsc = count_accuracy(W_true, W_thr)

            # plot weighted matrix to tensorboard
            # fig = plot_solution(W_true, W_new, graph_threshold)
            # self._writer.add_figure("graph", fig, iter)

            # log metrics
            logger.info(
                "Iteration {} - loss:{:.3e} mse:{:.3e} h:{:.3e} alpha:{:.2e} rho:{:.2e} fdr:{:.2%} tpr:{:.2%} shd:{:2d} nnz:{:2d} gsc:{:.2%}".format(
                    iter, loss, loss_mse, h_new, alpha, rho, fdr, tpr, shd, nnz, gsc
                )
            )

            # log metrics to tensorboard
            self._writer.add_scalar("loss", loss, iter)
            self._writer.add_scalar("h_est", h_new, iter)
            self._writer.add_scalar("fdr", fdr, iter)
            self._writer.add_scalar("tpr", tpr, iter)
            self._writer.add_scalar("shd", shd, iter)
            self._writer.add_scalar("gsc", gsc, iter)

            # save model parameters at each iteration
            pathfile = "{}/checkpoint_{:03d}.pt".format(self._output_dir, iter)
            state = {
                'model': model.state_dict(),
                'data': X.data.cpu(),
                'alpha': alpha,
                'rho': rho
            }
            torch.save(state, pathfile)

            # update alpha
            alpha += rho * h_new
            W_prev, h_prev = W_new, h_new

            # perform early stopping
            if h_new <= self._min_h and iter > self._min_iters:
                logger.info("Stop at {}-th iteration".format(iter))
                break

        # record time consume
        end_time = time.time()

        # compute metrics
        W_thr = post_process(W_prev, graph_threshold)
        fdr, tpr, fpr, shd, nnz, gsc = count_accuracy(W_true, W_thr)

        # show final metrics
        logger.info(
            "==================== Training Time: {:.1f} ====================".format(
                end_time - start_time
            )
        )
        logger.info(
            "FDR:{:.2%} TPR:{:.2%} FPR:{:.2%} SHD:{:2d} NNZ:{:2d} GSC:{:.2%}".format(
                fdr, tpr, fpr, shd, nnz, gsc
            )
        )

        # save estimated graph and true graph
        np.save("{}/graph_estimated.npy".format(self._output_dir), W_prev)
        np.save("{}/graph_true.npy".format(self._output_dir), W_true)

        # plot weighted matrix and save to file
        # fig = plot_solution(W_true, W_prev, graph_threshold)
        # imgfile = "{}/graph.svg".format(self._output_dir)
        # fig.savefig(imgfile, transparent=True)

    def train_epochs(self, model, X, optimizer, epochs, alpha, rho):
        """A basic training unit under certain alpha and rho

        Args:
            model (nn.Module): model to be trained
            X (Variable): input data
            optimizer (optim.Optimizer): optimizer for parameters
            scheduler (optim.lr_scheduler): learning rate scheduler
            epochs (int): number of runs for training with certain alpha and rho
            alpha (float): lagrange multiplier
            rho (float): penalty parameter

        Returns:
            loss: augmented lagrangian loss
            loss_mse: reconstruction loss
            h: acyclicity constraint error
            W: estimated weighted matrix
        """
        # record parameters
        loss, loss_mse, h, W = None, None, None, None

        for epoch in range(1, epochs + 1):
            # empty temporary parameter grad
            optimizer.zero_grad()

            # run model
            loss, loss_mse, loss_sparsity, h, W = model(X, alpha, rho)

            # compute loss backward grad
            loss.backward()

            # optimize parameters
            optimizer.step()

            # log metrics
            # logger.debug(
            #     "Epoch {:3d} - loss:{:.2e} mse:{:.2e} l1:{:.2e} dag:{:.2e} rho:{:.2e} alpha:{:.2e}".format(
            #         epoch,
            #         loss.item(),
            #         loss_mse.item(),
            #         loss_sparsity.item(),
            #         h.item(),
            #         rho,
            #         alpha,
            #     )
            # )

            # for test only
            W_est = W.data.cpu().numpy()
            W_thr = post_process(refine_distribution(W_est), self._thr)
            fdr, tpr, fpr, shd, nnz, gsc = count_accuracy(self._W_true, W_thr)
            self._i += 1
            logger.debug(
                "Epoch {:3d} - {:4d} loss:{:.2e} mse:{:.2e} l1:{:.2e} h:{:.2e} alpha:{:.2e} rho:{:.2e} | FDR:{:.2%} TPR:{:.2%} FPR:{:.2%} SHD:{:2d} NNZ:{:2d} GSC:{:.2%}".format(
                    epoch,
                    self._i,
                    loss.item(),
                    loss_mse.item(),
                    loss_sparsity.item(),
                    h.item(),
                    alpha,
                    rho,
                    fdr,
                    tpr,
                    fpr,
                    shd,
                    nnz,
                    gsc
                )
            )

            scalar_dict = {
                'loss': loss.item(),
                'mse': loss_mse.item(),
                'l1': loss_sparsity.item(),
                'h': h.item(),
                'alpha': alpha,
                'rho': rho,
                'fdr': fdr,
                'tpr': tpr,
                'fpr': fpr,
                'shd': shd,
                'nnz': nnz,
                'gsc': gsc,
            }
            self._writer.add_scalars('DEBUG', scalar_dict, self._i)

            # if self._i % 100 == 0:
            #     # fig = plot_graph([self._W_true, W_est, W_thr], ["True", "Estimated", "Thresholded"])
            #     # self._writer.add_figure("graph", fig, self._i)
            #     np.save("{}/w_est_{:04d}.npy".format(self._output_dir, self._i), W_est)
            #     np.save("{}/lat1_{:04d}.npy".format(self._output_dir, self._i), model.h1.data.cpu().numpy())
            #     np.save("{}/lat2_{:04d}.npy".format(self._output_dir, self._i), model.h2.data.cpu().numpy())
            #     np.save("{}/x_hat_{:04d}.npy".format(self._output_dir, self._i), model.x_hat.data.cpu().numpy())

        return loss.item(), loss_mse.item(), h.item(), refine_distribution(W_est)
