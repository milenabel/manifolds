import torch
from mlp import MLP
import torch.nn as nn

# Branch Network
class BranchNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False, use_layernorm=False):
        super(BranchNet, self).__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

# Trunk Network
class TrunkNet(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', spectral_norm=False, use_layernorm=False):
        super(TrunkNet, self).__init__([input_dim] + hidden_dims + [output_dim], activation, spectral_norm, use_layernorm)

# DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_encoded, trunk_input):
        """
        Inputs:
            branch_encoded: (latent_dim,) tensor (already encoded f)
            trunk_input: (batch_size, trunk_input_dim)
        """
        # Expand the encoded forcing vector to match batch size
        branch_output = self.branch_net(branch_encoded.unsqueeze(0).expand(trunk_input.shape[0], -1))
        trunk_output = self.trunk_net(trunk_input)
        result = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)
        return result
