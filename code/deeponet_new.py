import torch
import torch.nn as nn
from mlp import MLP

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
                 input_dim_branch1,    # full f input
                 input_dim_branch2,    # normal vector
                 input_dim_trunk,      # (x, y, z)
                 hidden_dims,          # e.g., [128, 128]
                 output_dim,           # typically 1
                 activation='relu',
                 trunk_activation='tanh',
                 spectral_norm=False):
        super().__init__()

        self.branch_func = BranchFuncNet(
            input_dim=input_dim_branch1,
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

    def forward(self, f_encoded, normals, x_points):
        B1 = self.branch_func(f_encoded)   # (N, p)
        B2 = self.branch_norm(normals)     # (N, p)
        T  = self.trunk(x_points)          # (N, p)

        combined = (B1 + B2) * T
        return torch.sum(combined, dim=1, keepdim=True)  # (N, 1)
