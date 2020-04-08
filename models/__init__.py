# Copyright (c) Yiwen Shao

# Apache 2.0

from .tdnn import TDNN
from .rnn import RNN


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              kernel_sizes=None, strides=None, dilations=None, bidirectional=True, dropout=0):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'TDNN':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN(in_dim, out_dim, num_layers,
                     hidden_dims, kernel_sizes, strides, dilations, dropout)

    else:
        model = RNN(in_dim, out_dim, num_layers, hidden_dims, bidirectional)

    return model
