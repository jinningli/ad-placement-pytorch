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

class ImpressionDataset(BaseDataset):
    def __init__(self):
        super(ImpressionDataset).__init__()
        self.data = []

    def initialize(self, opt):
        self.opt = opt
        fin = open(join(opt.dataroot, opt.phase + '_impression.txt'), 'r')
        print('Initializing Dataset...')
        assert(opt.batchSize == 1)
        cnt = 0
        item = None
        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            split = line.split('|')
            id = int(split[0].strip())
            if len(split) == 4:
                if item is not None:
                    self.data.append(item)
                item = {}
                item['id'] = id
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
                item['propensity'] = propensity

                features = split[3].lstrip('f ').strip()
                f0, f1, idx, val = self.parse_features(features)

                feature = get_sparse_tensor(idx, val)
                item['feature'] = [feature]
                label = torch.from_numpy(np.array([label], dtype='float32'))
                item['label'] = label

            if len(split) == 2:
                assert (id == item['id'])
                features = split[1].lstrip('f ').strip()
                f0, f1, idx, val = self.parse_features(features)
                feature = get_sparse_tensor(idx, val)
                item['feature'].append(feature)
        print(cnt)
        self.data.append(item)
        fin.close()

    def __getitem__(self, index):
        # store in sparse, get in dense
        item = self.data[index].copy()
        for k in range(len(item['feature'])):
            item['feature'][k] = item['feature'][k].to_dense().view(-1).unsqueeze(0)
        item['feature'] = torch.cat(item['feature'], dim=0)
        return item

    def __len__(self):
        return len(self.data)

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
