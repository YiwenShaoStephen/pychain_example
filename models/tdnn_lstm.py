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


class TDNNLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations,
                 bidirectional=False, dropout=0):
        super(TDNNLSTM, self).__init__()
        self.num_tdnn_layers = len(hidden_dims)
        self.num_lstm_layers = num_layers - len(hidden_dims)
        assert len(kernel_sizes) == self.num_tdnn_layers
        assert len(strides) == self.num_tdnn_layers
        assert len(dilations) == self.num_tdnn_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        # set lstm hidden_dim to the num_channels of the last cnn layer
        self.lstm_dim = hidden_dims[-1]
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(self.num_tdnn_layers)
        ])
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, self.num_lstm_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout)
        self.final_layer = nn.Linear(
            self.lstm_dim * self.num_directions, out_dim)

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        for i in range(len(self.tdnn)):
            # apply Tdnn
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        bsz = x.size(0)
        state_size = self.num_directions * \
            self.num_lstm_layers, bsz, self.lstm_dim
        h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True)  # (B, T, D)
        x, _ = self.lstm(x, (h0, c0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)  # (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


if __name__ == "__main__":
    num_layers = 2
    hidden_dim = 20
    in_dim = 10
    out_dim = 5
    net = TDNNLSTM(in_dim, out_dim, num_layers, hidden_dim)
    print(net)
    input = torch.randn(2, 30, 10)
    input_lengths = [30, 20]
    output, output_lengths = net(input, input_lengths)
    print(output.size())
    print(output_lengths)
