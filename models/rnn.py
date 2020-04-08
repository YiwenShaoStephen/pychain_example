# Copyright (c) Yiwen Shao

# Apache 2.0

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, rnn_type='LSTM', bidirectional=False):
        super(RNN, self).__init__()
        assert num_layers == len(hidden_dims)
        valid_rnn_types = ['LSTM', 'RNN', 'GRU']
        if rnn_type not in valid_rnn_types:
            raise ValueError("Only {0} types are supported but given {1}".format(
                valid_rnn_types, rnn_type))
        else:
            if rnn_type == 'LSTM':
                self.rnn_module = nn.LSTM
            if rnn_type == 'RNN':
                self.rnn_module = nn.RNN
            if rnn_type == 'GRU':
                self.rnn_module = nn.GRU
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn_layers = self._make_layer(in_dim, hidden_dims)
        self.final_layer = nn.Linear(hidden_dims[-1], out_dim)

    def _make_layer(self, in_dim, hidden_dims):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            layers.append(self.rnn_module(input_dim, hidden_dims[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(0, 1)  # (N, T, F) to (T, N, F)
        for rnn in self.rnn_layers:
            x, _ = rnn(x)
        if self.bidirectional:
            # (T, N, F*2) -> (T, N, F) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2)
        x = x.transpose(0, 1)  # (T, N, F) -> (N, T, F)
        x = self.final_layer(x)
        x = x.contiguous()
        return x


if __name__ == "__main__":
    num_layers = 2
    hidden_dims = [20, 30]
    in_dim = 10
    out_dim = 5
    net = RNN(in_dim, out_dim, num_layers, hidden_dims)
    print(net)
    input = torch.randn(30, 4, 10)
    output = net(input)
    print(output.size())
