# Copyright (c) Yiwen Shao

# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class tdnn_bn_relu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(tdnn_bn_relu, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = dilation * (kernel_size - 1) // 2
        self.dilation = dilation
        self.tdnn = nn.Conv1d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) +
            self.stride - 1
        ) // self.stride
        return out_lengths

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = self.bn(x)
        x = self.relu(x)
        x_lengths = self.output_lengths(x_lengths)
        return x, x_lengths


class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout=0):
        super(TDNN, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers
        assert len(dilations) == num_layers
        self.dropout = dropout
        self.num_layers = num_layers
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dims[-1], out_dim, True)

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        for i in range(len(self.tdnn)):
            # apply Tdnn
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


if __name__ == "__main__":
    kernel_size = 3
    dilation = 2
    num_layers = 1
    hidden_dims = [20]
    kernel_sizes = [3]
    strides = [2]
    dilations = [2]
    in_dim = 10
    out_dim = 5
    net = TDNN(in_dim, out_dim, num_layers,
               hidden_dims, kernel_sizes, strides, dilations)
    print(net)
    input = torch.randn(2, 8, 10)
    input_lengths = torch.IntTensor([8, 6])
    output, output_lengths = net(input, input_lengths)
    print(output.size())
    print(output_lengths)
