#!/usr/bin/env python3
# Copyright (c) Yiwen Shao

# Apache 2.0

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from dataset import ChainDataset, AudioDataLoader, BucketSampler
from models import get_model
from pychain.loss import ChainLoss
from pychain.graph import ChainGraph
import simplefst

parser = argparse.ArgumentParser(description='PyChain training')
# Datasets
parser.add_argument('--train', type=str, required=True,
                    help='training set json file')
parser.add_argument('--valid', type=str, required=True,
                    help='valid set json file')
parser.add_argument('--den-fst', type=str, required=True,
                    help='denominator fst path')
# Optimization options
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-bsz', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--valid-bsz', default=128, type=int, metavar='N',
                    help='valid batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer type')
parser.add_argument('--scheduler', type=str, default='plateau',
                    help='Learning rate scheduler')
parser.add_argument('--milestones', type=int, nargs='+', default=[5, 10],
                    help='Decrease learning rate at these epochs.(only for step decay)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '--pf', default=10, type=int,
                    help='print frequency')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='adam beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='adam beta2')
parser.add_argument('--curriculum', default=-1, type=int,
                    help='curriculum learning epochs that will start from short sequences')
# Checkpoints
parser.add_argument('--exp', default='exp/tdnn', type=str, metavar='PATH',
                    help='path to save checkpoint and log (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='TDNN',
                    choices=['TDNN', 'RNN', 'LSTM', 'GRU', 'TDNN-LSTM'],
                    help='model architecture: ')
parser.add_argument('--layers', default=5, type=int, help='number of layers')
parser.add_argument('--feat-dim', default=40, type=int,
                    help='number of features for each frame')
parser.add_argument('--hidden-dims', default=[256, 256, 256, 256, 256], type=int, nargs='+',
                    help='output dimensions for each hidden layer')
parser.add_argument('--num-targets', default=100, type=int,
                    help='number of nnet output dimensions (i.e. number of pdf-ids)')
parser.add_argument('--kernel-sizes', default=[3, 3, 3, 3, 3], type=int, nargs='+',
                    help='kernel sizes of TDNN/CNN layers (only required for TDNN)')
parser.add_argument('--dilations', default=[1, 1, 3, 3, 3], type=int, nargs='+',
                    help='dilations for TDNN/CNN kernels (only required for TDNN)')
parser.add_argument('--strides', default=[1, 1, 1, 1, 3], type=int, nargs='+',
                    help='strides for TDNN/CNN kernels (only required for TDNN)')
parser.add_argument('--bidirectional', default=False, type=bool,
                    help='bidirectional rnn')
# Miscs
parser.add_argument('--seed', type=int, default=0, help='manual seed')

args = parser.parse_args()
print(args)

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

best_loss = 1000  # best valid loss


def main():
    global best_loss
    writer = SummaryWriter(args.exp)
    print('Saving model and logs to {}'.format(args.exp))
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    trainset = ChainDataset(args.train)
    trainsampler = BucketSampler(trainset, args.train_bsz)
    trainloader = AudioDataLoader(trainset, batch_sampler=trainsampler)

    validset = ChainDataset(args.valid)
    validloader = AudioDataLoader(validset, batch_size=args.valid_bsz)

    # Model
    print("==> creating model '{}'".format(args.arch))
    model = get_model(args.feat_dim, args.num_targets, args.layers, args.hidden_dims, args.arch,
                      kernel_sizes=args.kernel_sizes, dilations=args.dilations,
                      strides=args.strides,
                      bidirectional=args.bidirectional,
                      dropout=args.dropout)
    print(model)

    if use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel()
                                           for p in model.parameters()) / 1000000.0))

    # loss
    den_fst = simplefst.StdVectorFst.read(args.den_fst)
    den_graph = ChainGraph(den_fst)
    criterion = ChainLoss(den_graph)

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # learning rate scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=start_epoch - 1)
    elif args.scheduler == 'exp':
        gamma = args.gamma ** (1.0 / args.epochs)  # final_lr = init_lr * gamma
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma, last_epoch=start_epoch - 1)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.gamma, patience=1)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.curriculum:
            trainsampler.shuffle(epoch)
        train_loss = train(
            trainloader, model, criterion, optimizer, writer, epoch, use_cuda)
        valid_loss = test(
            validloader, model, criterion, writer, epoch, use_cuda)

        # save model
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': valid_loss,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best, exp=args.exp)

        scheduler.step(valid_loss)

    print('Best loss:')
    print(best_loss)


def train(trainloader, model, criterion, optimizer, writer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, epoch)
    for batch_idx, (inputs, input_lengths, utt_ids, graphs) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()

        # compute output
        outputs, output_lengths = model(inputs, input_lengths)
        loss = criterion(outputs, output_lengths, graphs)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.detach().item(), output_lengths.sum())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if batch_idx % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Lr: {lr}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, batch_idx, len(trainloader), lr=lr, batch_time=batch_time,
                      loss=losses))
    # log to TensorBoard
    writer.add_scalar('train_loss', losses.avg, epoch)

    return losses.avg


def test(testloader, model, criterion, writer, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, input_lengths, utt_ids, graphs) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()

        # compute output
        outputs, output_lengths = model(inputs, input_lengths)
        loss = criterion(outputs, output_lengths, graphs)

        # measure accuracy and record loss
        losses.update(loss.detach().item(), output_lengths.sum())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if batch_idx % 1 == 0:  # print each batch stats since validset is small
            print('Validation: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, batch_idx, len(testloader), batch_time=batch_time,
                      loss=losses))
    # log to TensorBoard
    writer.add_scalar('valid_loss', losses.avg, epoch)

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, exp='exp', filename='checkpoint.pth.tar'):
    filepath = os.path.join(exp, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            exp, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
