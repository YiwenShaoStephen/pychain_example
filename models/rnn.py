# Copyright (c) Yiwen Shao

# Apache 2.0

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dim, rnn_type='LSTM', bidirectional=False, dropout=0):
        super(RNN, self).__init__()
        valid_rnn_types = ['LSTM', 'RNN', 'GRU']
        if rnn_type not in valid_rnn_types:
            raise ValueError("Only {0} types are supported but given {1}".format(
                valid_rnn_types, rnn_type))
        else:
            self.rnn_type = rnn_type
            if rnn_type == 'LSTM':
                self.rnn_module = nn.LSTM
            if rnn_type == 'RNN':
                self.rnn_module = nn.RNN
            if rnn_type == 'GRU':
                self.rnn_module = nn.GRU
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn_layer = self.rnn_module(self.in_dim, self.hidden_dim, self.num_layers,
                                         batch_first=True, bidirectional=bidirectional,
                                         dropout=dropout)
        self.final_layer = nn.Linear(hidden_dim * self.num_directions, out_dim)

    def forward(self, x, x_lengths):
        bsz = x.size(0)
        state_size = self.num_directions * \
            self.num_layers, bsz, self.hidden_dim
        h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)
        if self.rnn_type == 'LSTM':
            h0 = (h0, c0)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True)  # (B, T, D)
        x, _ = self.rnn_layer(x, h0)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)  # (B, T, D)
        x = self.final_layer(x)
        return x, x_lengths


if __name__ == "__main__":
    num_layers = 2
    hidden_dim = 20
    in_dim = 10
    out_dim = 5
    net = RNN(in_dim, out_dim, num_layers, hidden_dim)
    print(net)
    input = torch.randn(2, 30, 10)
    input_lengths = [30, 20]
    output, output_lengths = net(input, input_lengths)
    print(output.size())
    print(output_lengths)
