# deeponet_vanilla.py

import torch
import torch.nn as nn
from mlp import MLP

# Defining the Branch Network
class BranchNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation=activation, spectral_norm=spectral_norm, use_layernorm=use_layernorm)

# Defining the Trunk Network
class TrunkNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', spectral_norm=False, use_layernorm=False):
        super().__init__([input_dim] + hidden_dims + [output_dim], activation=activation, spectral_norm=spectral_norm, use_layernorm=use_layernorm)

# Defining the DeepONet using BranchNet and TrunkNet
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)  # (batch_size, output_dim)
        trunk_output = self.trunk_net(trunk_input)      # (batch_size, output_dim)

        result = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # (batch_size, 1)
        return result
