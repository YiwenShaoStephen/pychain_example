# Copyright 2019 Yiwen Shao

# Apache 2.0

from .tdnn import TDNN
from .rnn import RNN


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              kernel_sizes=None, dilations=None, bidirectional=True):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'TDNN':
        if not kernel_sizes or not dilations:
            raise ValueError(
                'Please specify kernel sizes and their dilations for TDNN')
        model = TDNN(in_dim, out_dim, num_layers,
                     hidden_dims, kernel_sizes, dilations)

    else:
        model = RNN(in_dim, out_dim, num_layers, hidden_dims, bidirectional)

    return model
