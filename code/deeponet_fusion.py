import torch
import torch.nn as nn
from mlp import MLP

# Branch Networks
class BranchNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

class TrunkNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

# Fusion MLP
class FusionMLP(MLP):
    def __init__(self, input_dim=2, hidden_dims=[32, 32], output_dim=1, activation='relu', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

# Full DeepONet (Fusion Version)
class DeepONetFusion(nn.Module):
    def __init__(self, branch_f_net, branch_n_net, trunk_net, fusion_mlp):
        super().__init__()
        self.branch_f_net = branch_f_net
        self.branch_n_net = branch_n_net
        self.trunk_net = trunk_net
        self.fusion_mlp = fusion_mlp

    def forward(self, branch_f_input, branch_n_input, trunk_input):
        branch_f_out = self.branch_f_net(branch_f_input)
        branch_n_out = self.branch_n_net(branch_n_input)
        trunk_out = self.trunk_net(trunk_input)

        dot_f = torch.sum(branch_f_out * trunk_out, dim=-1, keepdim=True)
        dot_n = torch.sum(branch_n_out * trunk_out, dim=-1, keepdim=True)

        fusion_input = torch.cat([dot_f, dot_n], dim=1)  # Shape (batch_size, 2)
        output = self.fusion_mlp(fusion_input)           # Output shape (batch_size, 1)

        return output
