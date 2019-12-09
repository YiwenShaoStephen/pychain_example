import os
import re
import torch
import simplefst
import kaldi_io
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data import DataLoader
from pychain import ChainGraph, ChainGraphBatch


def _collate_fn(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[0].size(0), reverse=True)
    max_seqlength = batch[0][0].size(0)
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    graph_list = []
    max_num_transitions = 0
    max_num_states = 0
    for x in range(minibatch_size):
        sample = batch[x]
        feat, graph = sample
        feat_length = feat.size(0)
        feats[x].narrow(0, 0, feat_length).copy_(feat)
        graph_list.append(graph)
        if graph.num_transitions > max_num_transitions:
            max_num_transitions = graph.num_transitions
        if graph.num_states > max_num_states:
            max_num_states = graph.num_states
    num_graphs = ChainGraphBatch(
        graph_list, max_num_transitions=max_num_transitions, max_num_states=max_num_states)
    return feats, num_graphs


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class ChainDataset(data.Dataset):

    def __init__(self, feat_dir, fst_dir):
        super(ChainDataset, self).__init__()
        self.feat_dir = feat_dir
        self.fst_dir = fst_dir

        self.feat_scp = os.path.join(self.feat_dir, 'feats.scp')
        self.fst_scp = os.path.join(self.fst_dir, 'fst_nor.1.scp')
        self.feat_len_map = os.path.join(self.feat_dir, 'utt2featlen.txt')

        if not os.path.exists(self.feat_len_map):
            print('{} does not exist, generating utt2featlen.txt (a map from utt_id'
                  ' to feature length) for the first time... It is used to form a minibatch'
                  ' with similar length in training.'.format(self.feat_len_map))
            with open(self.feat_len_map, 'w') as map_f:
                with open(self.feat_scp) as f:
                    for i, line in tqdm(enumerate(f)):
                        utt_id, feat_ark = line.strip().split()
                        feat = kaldi_io.read_mat(feat_ark)
                        feat_len = feat.shape[0]
                        map_f.write('{} {}\n'.format(utt_id, feat_len))
        # Pairing
        self.samples = []  # list of dicts
        self.utt_ids = {}  # a dict that maps utt_ids(str) to id(int)

        self.samples_tmp = []
        with open(self.feat_scp) as f:
            for i, line in enumerate(f):
                utt_id, feat_ark = line.strip().split()
                self.utt_ids[utt_id] = i
                self.samples_tmp.append({'utt_id': utt_id, 'feat': feat_ark})

        with open(self.feat_len_map) as f:
            for i, line in enumerate(f):
                utt_id, feat_len = line.strip().split()
                id = self.utt_ids[utt_id]
                self.samples_tmp[id]['feat_len'] = int(feat_len)

        # we always cache all fsts into memory at once as its relatively small
        with open(self.fst_scp) as f:
            print("Loading training FSTs...")
            for i, line in tqdm(enumerate(f)):
                utt_id, fst_rxf = line.strip().split()
                if utt_id not in self.utt_ids:
                    raise ValueError(
                        '{} has no corresponding feats'.format(utt_id))
                id = self.utt_ids[utt_id]
                filename, offset = self.parse_rxfile(fst_rxf)
                filename = filename.split('/')[-1]
                file_path = os.path.join(self.fst_dir, filename)
                fst = simplefst.StdVectorFst.read_ark(file_path, offset)
                graph = ChainGraph(fst)
                dict_tmp = self.samples_tmp[id]
                dict_tmp['graph'] = graph
                self.samples.append(dict_tmp)
                #self.samples[id]['graph'] = graph

        # sort the samples by their feature length
        self.samples = sorted(
            self.samples, key=lambda sample: sample['feat_len'])

    def parse_rxfile(self, file):
        # separate offset from filename
        if re.search(':[0-9]+$', file):
            (file, offset) = file.rsplit(':', 1)
        return file, int(offset)

    def __getitem__(self, index):
        sample = self.samples[index]
        feat_ark = sample['feat']
        graph = sample['graph']
        feat = torch.from_numpy(kaldi_io.read_mat(feat_ark))
        return feat, graph

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from pychain.loss import ChainLoss

    feat_dir = '/export/b08/yshao/kaldi/egs/wsj/s5/data/train_si284_spe2e_hires'
    tree_dir = '/export/b08/yshao/kaldi/egs/wsj/s5/exp/chain/e2e_tree'
    trainset = ChainDataset(feat_dir, tree_dir)
    trainloader = AudioDataLoader(trainset, batch_size=2, shuffle=True)

    feat, graphs = next(iter(trainloader))
    print(feat.size())
    den_fst_path = '/export/b08/yshao/kaldi/egs/wsj/s5/exp/chain/e2e_tree/den.fst'
    den_fst = simplefst.StdVectorFst.read(den_fst_path)
    den_graph = ChainGraph(den_fst, initial='recursive')
    print(den_graph.num_states)
    den_graph_batch = ChainGraphBatch(den_graph, batch_size=2)
    criterion = ChainLoss(den_graph_batch)
    torch.manual_seed(1)
    nnet_output = torch.randn(2, 10, 100)  # (B, T, D)
    nnet_output.requires_grad = True

    obj = criterion(nnet_output, graphs)
    obj.backward()
    print(obj)
    print(nnet_output.grad)
