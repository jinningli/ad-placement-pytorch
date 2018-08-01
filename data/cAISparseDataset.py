from data.baseDataset import BaseDataset
from utils.utils import mkdirs
from os.path import join
import os
import scipy.sparse as sp
import numpy as np
import torch
import pickle
from utils.utils import to_csr
from utils.utils import get_sparse_tensor
import torch.sparse as sparse

# def fillto(nparr, dim=200):
#     length = len(nparr)
#     if length > dim:
#         print('Warning: length exceed limit: ' + str(length) + '/' + str(dim))
#         res = nparr[:dim]
#         return res, dim
#     else:
#         fillarr = np.zeros(dim-length, dtype='int64')
#         res = np.concatenate((nparr, fillarr), axis=0)
#         return res, length

class CAISparseDataset(BaseDataset):
    def __init__(self):
        super(CAISparseDataset).__init__()
        self.data = []

    def initialize(self, opt):
        self.opt = opt
        fin = open(join(opt.dataroot, opt.phase + '.txt'), 'r')
        print('Initializing Dataset...')
        if self.opt.cache:
            print('Warning: cache method not implemented for sparse dataset')
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            split = line.split('|')
            id = int(split[0].strip())
            if len(split) == 4:
                l = split[1]
                assert l.startswith('l')

                l = l.lstrip('l ').strip()
                if l == '0.999':
                    label = 0
                elif l == '0.001':
                    label = 1
                else:
                    raise Exception('Label not valid: ' + str(l))
                p = split[2]
                assert p.startswith('p')
                p = p.lstrip('p ').strip()
                propensity = float(p)
                propensity = torch.from_numpy(np.array([propensity], dtype='float32'))
                features = split[3].lstrip('f ').strip()
                f0, f1, idx, val = self.parse_features(features)
                label = torch.from_numpy(np.array([label], dtype='float32'))
                self.data.append({'p': propensity, 'idx': idx, 'val': val, 'label': label})
                """
                propensity: torch.FloatTensor [1]
                idx: list [k]
                val: list [k]
                label: torch.FloatTensor [1]
                """
        fin.close()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'Sparse CAI Dataset'

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

