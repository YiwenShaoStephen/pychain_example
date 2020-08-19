# Copyright (c) Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import OrderedDict
import json

from io import BytesIO
import librosa
from subprocess import run, PIPE
import torchaudio

import torch
import simplefst
import kaldi_io
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from pychain import ChainGraph, ChainGraphBatch


def parse_rxfile(file):
    # separate offset from filename
    if re.search(':[0-9]+$', file):
        (file, offset) = file.rsplit(':', 1)
    return file, int(offset)


def _collate_fn_train(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    graph_list = []
    utt_ids = []
    max_num_transitions = 0
    max_num_states = 0
    for i in range(minibatch_size):
        feat, length, utt_id, graph = batch[i]
        feats[i, :length, :].copy_(feat)
        utt_ids.append(utt_id)
        feat_lengths[i] = length
        graph_list.append(graph)
        if graph.num_transitions > max_num_transitions:
            max_num_transitions = graph.num_transitions
        if graph.num_states > max_num_states:
            max_num_states = graph.num_states
    num_graphs = ChainGraphBatch(
        graph_list, max_num_transitions=max_num_transitions, max_num_states=max_num_states)
    return feats, feat_lengths, utt_ids, num_graphs


def _collate_fn_test(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    for i in range(minibatch_size):
        feat, length, utt_id = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        utt_ids.append(utt_id)
    return feats, feat_lengths, utt_ids


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for ChainDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        if self.dataset.train:
            self.collate_fn = _collate_fn_train
        else:
            self.collate_fn = _collate_fn_test


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        codes from deepspeech.pytorch 
        https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        """
        super(BucketSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class ChainDataset(data.Dataset):

    def __init__(self, data_json_path, train=True, cache_graph=True, sort=True):
        super(ChainDataset, self).__init__()
        self.train = train
        self.cache_graph = cache_graph
        self.sort = sort
        self.samples = []  # list of dicts

        with open(data_json_path, 'rb') as f:
            loaded_json = json.load(f, object_pairs_hook=OrderedDict)

        print("Initializing dataset...")
        for utt_id, val in tqdm(loaded_json.items()):
            sample = {}
            sample['utt_id'] = utt_id
            sample['wav'] = val['wav']
            sample['text'] = val['text']
            sample['duration'] = float(val['duration'])
            sample['feat'] = val['feat']
            sample['length'] = int(val['length'])

            if self.train:  # only training data has fst (graph)
                fst_rxf = val['numerator_fst']
                if self.cache_graph:  # cache all fsts at once
                    filename, offset = parse_rxfile(fst_rxf)
                    fst = simplefst.StdVectorFst.read_ark(filename, offset)
                    graph = ChainGraph(fst, log_domain=True)
                    if graph.is_empty:
                        continue
                    sample['graph'] = graph
                else:
                    sample['graph'] = fst_rxf

            self.samples.append(sample)

        if self.sort:
            # sort the samples by their feature length
            self.samples = sorted(
                self.samples, key=lambda sample: sample['duration'])

    def __getitem__(self, index):
        sample = self.samples[index]
        utt_id = sample['utt_id']
        feat_ark = sample['feat']
        feat = torch.from_numpy(kaldi_io.read_mat(feat_ark))
        feat_length = sample['length']

        if self.train:
            if self.cache_graph:
                graph = sample['graph']
            else:
                fst_rxf = sample['graph']
                filename, offset = parse_rxfile(fst_rxf)
                fst = simplefst.StdVectorFst.read_ark(filename, offset)
                graph = ChainGraph(fst)
            return feat, feat_length, utt_id, graph
        else:
            utt_id = sample['utt_id']
            return feat, feat_length, utt_id

    def __len__(self):
        return len(self.samples)
