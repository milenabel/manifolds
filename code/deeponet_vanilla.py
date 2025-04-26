import torch
from mlp import MLP
import torch.nn as nn

# Defining the Branch Network
class BranchNet(MLP):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', spectral_norm=False):
        super(BranchNet, self).__init__([input_dim] + hidden_dim + [output_dim], activation, spectral_norm)

# Defining the Trunk Network
class TrunkNet(MLP):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', spectral_norm=False):
        super(TrunkNet, self).__init__([input_dim] + hidden_dim + [output_dim], activation, spectral_norm)

# Defining the DeepONet using the BranchNet and TrunkNet
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)  # (batch_size, output_dim)
        trunk_output = self.trunk_net(trunk_input)     # (num_points, output_dim)

        # Perform element-wise multiplication
        result = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # Ensure second dimension is retained

        # Now `result` should have shape [batch_size, 1]
        return result
    



