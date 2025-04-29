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

# Fusion2 MLP
class FusionMLP(MLP):
    def __init__(self, input_dim=256, hidden_dims=[128, 128], output_dim=128, activation='relu', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

# Full DeepONet (Fusion2 Version)
class DeepONetFusion2(nn.Module):
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

        fusion_input = torch.cat([branch_f_out, branch_n_out], dim=1)  # (batch_size, 256)
        fused_branch = self.fusion_mlp(fusion_input)                   # (batch_size, 128)

        output = torch.sum(fused_branch * trunk_out, dim=-1, keepdim=True)  # final output shape (batch_size, 1)
        return output
