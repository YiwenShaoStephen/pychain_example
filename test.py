#!/usr/bin/env python3
# Copyright (c) Yiwen Shao

# Apache 2.0

import argparse
import os

import torch
import torch.nn.parallel

from dataset import ChainDataset, AudioDataLoader
from models import get_model

import kaldi_io

parser = argparse.ArgumentParser(description='PyChain test')
# Datasets
parser.add_argument('--test', type=str, required=True,
                    help='test set json file')
# Model
parser.add_argument('--exp', default='exp/tdnn',
                    type=str, metavar='PATH', required=True,
                    help='dir to load model and save output')
parser.add_argument('--model', default='model_best.pth.tar', type=str,
                    help='model checkpoint')
parser.add_argument('--results', default='posteriors.ark', type=str,
                    help='results filename')
parser.add_argument('--bsz', default=128, type=int,
                    help='test batchsize')

args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()


def main():
    # Data
    testset = ChainDataset(args.test, train=False)
    testloader = AudioDataLoader(testset, batch_size=args.bsz)

    # Model
    checkpoint_path = os.path.join(args.exp, args.model)
    with open(checkpoint_path, 'rb') as f:
        state = torch.load(f)
        model_args = state['args']
        print("==> creating model '{}'".format(model_args.arch))
        model = get_model(model_args.feat_dim, model_args.num_targets,
                          model_args.layers, model_args.hidden_dims,
                          model_args.arch, kernel_sizes=model_args.kernel_sizes,
                          dilations=model_args.dilations,
                          strides=model_args.strides,
                          bidirectional=model_args.bidirectional)
        print(model)

        if use_cuda:
            model = torch.nn.DataParallel(model).cuda()

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model.load_state_dict(state['state_dict'])

    output_file = os.path.join(args.exp, args.results)
    test(testloader, model, output_file, use_cuda)


def test(testloader, model, output_file, use_cuda):
    # switch to test mode
    model.eval()
    with open(output_file, 'wb') as f:
        for i, (inputs, input_lengths, utt_ids) in enumerate(testloader):
            lprobs, output_lengths = model(inputs, input_lengths)
            for j in range(inputs.size(0)):
                output_length = output_lengths[j]
                utt_id = utt_ids[j]
                kaldi_io.write_mat(
                    f, (lprobs[j, :output_length, :]).cpu().detach().numpy(), key=utt_id)


if __name__ == '__main__':
    main()
