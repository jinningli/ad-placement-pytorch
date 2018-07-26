from data.baseDataset import BaseDataset
from utils.utils import mkdirs
from os.path import join
import os
import scipy.sparse as sp
import numpy as np
import torch
import torch.sparse as sparse
from itertools import groupby
import pickle
from utils.utils import to_csr

# def save_csr()
#     sp.save_npz('tmp/X_train_sparse.npz', X_train, compressed=False)

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

    def initialize(self, opt):
        self.opt = opt
        fin = open(join(opt.dataroot, opt.phase + '.txt'), 'r')
        print('Initializing Dataset...')
        if os.path.exists(join(opt.dataroot, 'cache', opt.phase + '.pkl')):
            print('Loading dataset from cache')
            with open(join(opt.dataroot, 'cache', opt.phase + '.pkl'), 'rb') as cache:
                self.data = pickle.load(cache)
        else:
            if not os.path.exists(join(opt.dataroot, 'cache')):
                os.mkdir(join(opt.dataroot, 'cache'))
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
                feature = to_csr(idx, val)
                # feature = get_sparse_tensor(idx, val)
                self.data.append({'id': id, 'feature': feature})
            fin.close()
            with open(join(opt.dataroot, 'cache', opt.phase + '.pkl'), 'wb') as cache:
                print('Dump dataset into ' + join(opt.dataroot, 'cache', opt.phase + '.pkl'))
                pickle.dump(self.data, cache)

        # groups = groupby(self.data, key=lambda x: x['id'])
        # for id, group in groups:
        #     ls = []
        #     for item in group:
        #         ls.append(item)
        #     self.groups.append(ls)

    def __getitem__(self, index):
        # store in sparse, get in dense
        # group = self.groups[index]
        # cat = None
        # id = None
        # for item in group:
        #     if cat is None:
        #         cat = item['feature'].to_dense()
        #         id = item['id']
        #     else:
        #         cat = torch.cat((cat, item['feature'].to_dense()), dim=0)
        #
        item = self.data[index].copy()
        item['feature'] = torch.from_numpy(item['feature'].toarray().astype('float32')).view(-1)
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

