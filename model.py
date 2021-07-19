import torch
import torch.nn as nn
import numpy as np


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        permute=(0, 2, 1),
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self._p = permute
        self._bn = nn.BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, x):
        x = self._bn(x.permute(*self._p)).permute(*self._p)
        return x


class NonLinerTransformer(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer):
        """Component for encoder and decoder

        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(NonLinerTransformer, self).__init__()
        dims = (
            [(in_dim, n_hid)]
            + [(n_hid, n_hid) for _ in range(n_layer - 1)]
            + [(n_hid, out_dim)]
        )
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        # bn_layers = [BatchNorm(n_hid) for _ in range(n_layer)]
        lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            # layers.append(bn_layers[i])
            layers.append(lr_layers[i])
        layers.append(fc_layers[-1])
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class InvTransformer(nn.Module):
    def __init__(self, n_dim, n_hid, n_latent, n_layer):
        super(InvTransformer, self).__init__()
        self.augment = nn.Linear(n_dim, 2 * n_latent)
        self.reduce = nn.Linear(2 * n_latent, n_latent)
        self.s1 = NonLinerTransformer(n_latent, n_hid, n_latent, n_layer)
        self.s2 = NonLinerTransformer(n_latent, n_hid, n_latent, n_layer)
        self.t1 = NonLinerTransformer(n_latent, n_hid, n_latent, n_layer)
        self.t2 = NonLinerTransformer(n_latent, n_hid, n_latent, n_layer)

    def forward(self, x):
        x = self.augment(x)
        u1, u2 = torch.chunk(x, chunks=2, dim=-1)
        v1 = u1 * torch.exp(self.s2(u2)) + self.t2(u2)
        v2 = u2 * torch.exp(self.s1(v1)) + self.t1(v1)
        z = torch.cat((v1, v2), -1)
        z = self.reduce(z)
        return z


class CAE(nn.Module):
    def __init__(self, d, n_dim, n_hid, n_latent, n_layer, lambda_sparsity, psp=False):
        """
        Graph Autoencoder

        Code modified from:
            https://github.com/huawei-noah/trustworthyAI/blob/master/Causal_Structure_Learning/GAE_Causal_Structure_Learning/src/models/gae.py

        Args:
            d (int): variables number
            n_dim (int): features number, aka input dimension.
            n_hid (int): encoder and decoder layer dimension.
            n_latent (int): encoded latent layer dimension.
        """
        super(CAE, self).__init__()
        # constant
        self._d = d
        self._l1 = lambda_sparsity
        self._psp = psp

        # Non-linear transformer for data dimensions
        self.encoder = NonLinerTransformer(n_dim, n_hid, n_latent, n_layer)
        self.decoder = NonLinerTransformer(n_latent, n_hid, n_dim, n_layer)

        # Test: Non-linear invertible transformer for data dimensions
        # self.encoder = InvTransformer(n_dim, n_hid, n_latent, n_layer)
        # self.decoder = InvTransformer(n_latent, n_hid, n_dim, n_layer)

        # initial value of W has substantial impact on model performance
        _mask = torch.Tensor(1 - np.eye(d))
        self.register_buffer("_mask", _mask)
        self._W_pre = nn.Parameter(
            torch.Tensor(np.random.uniform(low=-0.1, high=0.1, size=(d, d)))
        )

    def forward(self, x, alpha, rho):
        # constant
        n = x.shape[0]

        # get weighted matrix parameter
        # https://github.com/huawei-noah/trustworthyAI/issues/21#event-4920062931
        self.W = self._W_pre * self._mask

        # model forward
        self.h1 = self.encoder(x)
        self.h2 = torch.einsum("ijk,jl->ilk", self.h1, self.W)
        self.x_hat = self.decoder(self.h2)

        # compute loss
        loss_mse = torch.square(torch.norm(self.x_hat - x))
        loss_sparsity = self._compute_l1(self.W)
        h = torch.trace(torch.matrix_exp(self.W * self.W)) - self._d
        loss = (
            0.5 / n * loss_mse
            + self._l1 * loss_sparsity
            + alpha * h
            + 0.5 * rho * h * h
        )

        # for debug and watch
        # mask = torch.Tensor(list((0, 1) + (0,) * 98)).cuda()
        # xmsk = x_hat.squeeze().mean(dim=0)
        # xhc = (x_hat.squeeze() * (1 - mask) + xmsk * mask).unsqueeze(-1)
        # lmc = 0.5 / n * torch.square(torch.norm(xhc - x))

        return loss, loss_mse, loss_sparsity, h, self.W

    def _compute_l1(self, W):
        if self._psp:
            W = W / torch.max(torch.abs(W))
            loss = 2 * torch.sigmoid(2 * W) - 1
            return loss.norm(p=1) / self._d ** 2
        else:
            return W.norm(p=1)


if __name__ == "__main__":
    model = CAE(d=100, n_dim=1, n_hid=16, n_latent=1, n_layer=3, lambda_sparsity=1.0)
    print(model, end="\n\n")
    for name, parameters in model.named_parameters():
        print(name, ":", parameters.size())
