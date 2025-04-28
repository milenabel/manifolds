import torch
import torch.nn as nn
from mlp import MLP

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(BranchNet, self).__init__()
        self.network = MLP([input_dim] + hidden_dims + [output_dim])

    def forward(self, x):
        return self.network(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(TrunkNet, self).__init__()
        self.network = MLP([input_dim] + hidden_dims + [output_dim])

    def forward(self, x):
        return self.network(x)

class DeepONetDualBranch(nn.Module):
    def __init__(self, branch_f_net, branch_n_net, trunk_net):
        super().__init__()
        self.branch_f_net = branch_f_net
        self.branch_n_net = branch_n_net
        self.trunk_net = trunk_net

    def forward(self, branch_f_input, branch_n_input, trunk_input):
        branch_f_out = self.branch_f_net(branch_f_input)
        branch_n_out = self.branch_n_net(branch_n_input)
        trunk_out = self.trunk_net(trunk_input)

        z = branch_f_out * branch_n_out * trunk_out
        output = torch.sum(z, dim=1, keepdim=True)   
        return output

