from data.baseDataset import BaseDataset
from utils.utils import mkdirs
from os.path import join
import os
import scipy.sparse as sp
import numpy as np
import torch
import torch.sparse as sparse
from itertools import groupby

def get_sparse_tensor(idxs, vals, dim=74000):
    rows = torch.from_numpy(np.zeros(len(idxs), dtype='int64')).view(1, -1)
    cols = torch.from_numpy(np.array(idxs, dtype='int64')).view(1, -1)
    i = torch.cat((rows, cols), dim=0)
    vals = np.ones(len(idxs), dtype='int64') #TODO use value?
    v = torch.from_numpy(np.array(vals, dtype='float32'))
    sparse_matrix = sparse.FloatTensor(i, v, torch.Size([1, dim]))
    # print(sparse_matrix)
    return sparse_matrix

def fillto(nparr, dim=200):
    length = len(nparr)
    if length > dim:
        print('Warning: length exceed limit: ' + str(length) + '/' + str(dim))
        res = nparr[:dim]
        return res, dim
    else:
        fillarr = np.zeros(dim-length, dtype='int64')
        res = np.concatenate((nparr, fillarr), axis=0)
        return res, length

class CAITestDataset(BaseDataset):
    def __init__(self):
        super(CAITestDataset).__init__()
        self.data = []
        self.groups = []

    def initialize(self, opt):
        self.opt = opt
        fin = open(join(opt.dataroot, opt.phase + '.txt'), 'r')
        print('Initializing Dataset...')
        cnt = 0

        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)

            split = line.split('|')
            id = int(split[0].strip())

            assert len(split) == 2

            features = split[1].lstrip('f ').strip()

            f0, f1, idx, val = self.parse_features(features)
            feature = get_sparse_tensor(idx, val)

            self.data.append({'id': id, 'feature': feature})

        fin.close()

        groups = groupby(self.data, key=lambda x: x['id'])
        for id, group in groups:
            ls = []
            for item in group:
                ls.append(item)
            self.groups.append(ls)

    def __getitem__(self, index):
        # store in sparse, get in dense
        group = self.groups[index]
        cat = None
        id = None
        for item in group:
            if cat is None:
                cat = item['feature'].to_dense()
                id = item['id']
            else:
                cat = torch.cat((cat, item['feature'].to_dense()), dim=0)
        return {'id': id, 'feature': cat}

    def __len__(self):
        return len(self.groups)

    def name(self):
        return 'General CAI Dataset'

    def parse_features(self, s):
        split = s.split(' ')
        f0 = split[0]
        assert f0.startswith('0:')
        f0 = int(f0[2:])

        f1 = split[1]
        assert f1.startswith('1:')
        f1 = int(f1[2:])

        idx = []
        values = []

        for fv in split[2:]:
            f, v = fv.split(':')
            idx.append(int(f) - 2)
            values.append(int(v))

        return f0, f1, idx, values

