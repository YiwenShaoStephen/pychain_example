# Copyright (c) Yiwen Shao

# Apache 2.0

from .tdnn import TDNN
from .rnn import RNN
from .tdnn_lstm import TDNNLSTM


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              kernel_sizes=None, strides=None, dilations=None, bidirectional=True, dropout=0):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU', 'TDNN-LSTM']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'TDNN':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN(in_dim, out_dim, num_layers,
                     hidden_dims, kernel_sizes, strides, dilations, dropout)

    elif arch == 'TDNN-LSTM':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN-LSTM')
        model = TDNNLSTM(in_dim, out_dim, num_layers, hidden_dims, kernel_sizes,
                         strides, dilations, bidirectional, dropout)
    else:
        # we simply use same hidden dim for all rnn layers
        hidden_dim = hidden_dims[0]
        model = RNN(in_dim, out_dim, num_layers, hidden_dim,
                    arch, bidirectional, dropout)

    return model
