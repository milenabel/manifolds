# deeponet_new.py

import torch
import torch.nn as nn
from mlp import MLP
from encoder import SimpleEncoder  # import our encoder

class BranchFuncNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm)

class BranchNormNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm)

class TrunkNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', spectral_norm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm)

class DeepONetDualBranch(nn.Module):
    def __init__(self, 
                 input_dim_branch1,    # full f dimension (before encoding)
                 input_dim_branch2,    # normals dimension (3)
                 input_dim_trunk,      # spatial coords (3)
                 hidden_dims,          # MLP hidden dims
                 output_dim,           # 1
                 activation='relu',
                 trunk_activation='tanh',
                 spectral_norm=False,
                 encoder_latent_dim=128):

        super().__init__()

        # === New Encoder ===
        self.encoder_f = SimpleEncoder(input_dim=input_dim_branch1, latent_dim=encoder_latent_dim)

        # Branch function input now uses latent_dim not full input_dim_branch1
        self.branch_func = BranchFuncNet(
            input_dim=encoder_latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            spectral_norm=spectral_norm
        )

        self.branch_norm = BranchNormNet(
            input_dim=input_dim_branch2,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            spectral_norm=spectral_norm
        )

        self.trunk = TrunkNet(
            input_dim=input_dim_trunk,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=trunk_activation,
            spectral_norm=spectral_norm
        )

    def forward(self, f_full, normals, x_points):
        """
        f_full: (N_pts, input_dim_branch1) (still full f tensor expanded per point)
        normals: (N_pts, 3)
        x_points: (N_pts, 3)
        """

        # 1. Encode f once
        encoded_f = self.encoder_f(f_full[0:1, :])  # Pick only one (all rows identical)

        # 2. Expand encoded f lightweightly
        branch_input_f = encoded_f.expand(x_points.shape[0], -1)  # (N_pts, latent_dim)

        # 3. Forward through branches
        B1 = self.branch_func(branch_input_f)
        B2 = self.branch_norm(normals)
        T  = self.trunk(x_points)

        combined = (B1 + B2) * T
        return torch.sum(combined, dim=1, keepdim=True)  # (N_pts, 1)
