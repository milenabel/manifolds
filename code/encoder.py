# encoder.py

import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256], activation='relu'):
        super(SimpleEncoder, self).__init__()

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }

        self.activation = activations.get(activation.lower(), nn.ReLU())

        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(self.activation)

        self.network = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, f_tensor):
        return self.network(f_tensor)
