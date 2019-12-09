# Copyright 2019 Yiwen Shao

# Apache 2.0

import torch
import torch.nn as nn


class tdnn_bn_relu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation=1):
        super(tdnn_bn_relu, self).__init__()
        self.tdnn = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, dilations):
        super(TDNN, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        self.num_layers = num_layers
        self.tdnn_layers = self._make_layer(
            in_dim, hidden_dims, kernel_sizes, dilations)
        self.final_layer = nn.Conv1d(hidden_dims[-1], out_dim, kernel_size=1)

    def _make_layer(self, in_dim, hidden_dims, kernel_sizes, dilations):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            layers.append(tdnn_bn_relu(
                input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.size()) == 3  # x is of size (N, T, F)
        # turn x to (N, F, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        x = self.tdnn_layers(x)
        x = self.final_layer(x)
        x = x.transpose(2, 1).contiguous()  # turn it back to (N, T, F)
        return x


if __name__ == "__main__":
    kernel_size = 3
    dilation = 2
    num_layers = 1
    hidden_dims = [20]
    kernel_sizes = [3]
    dilations = [2]
    in_dim = 10
    out_dim = 5
    net = TDNN(in_dim, out_dim, num_layers,
               hidden_dims, kernel_sizes, dilations)
    print(net)
    input = torch.randn(2, 8, 10)
    output = net(input)
    print(output.size())
