import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu', spectral_norm=False, use_layernorm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = self.get_activation(activation)
        self.use_layernorm = use_layernorm
        self.spectral_norm = spectral_norm
        self.activation_name = activation.lower()

        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            if spectral_norm:
                linear_layer = nn.utils.spectral_norm(linear_layer)

            self.init_weights(linear_layer)
            self.layers.append(linear_layer)

            if use_layernorm and i < len(layer_sizes) - 2:
                self.layers.append(nn.LayerNorm(layer_sizes[i + 1]))

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            if self.activation_name == 'relu' or self.activation_name == 'leaky_relu':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif self.activation_name == 'tanh':
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:  # if it's LayerNorm
                x = layer(x)
        return self.layers[-1](x)  # Last Linear layer without activation

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
