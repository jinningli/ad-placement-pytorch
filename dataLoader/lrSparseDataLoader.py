from dataLoader.baseDataLoader import BaseDataLoader
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.sparse as sparse

class LRSparseDataLoader(BaseDataLoader):
    def name(self):
        return 'LRSparseDataLoader'

    def initialize(self, dataset, opt):
        BaseDataLoader.initialize(self, dataset, opt)
        self.dataset = dataset

        if opt.isTrain:
            self.shuffle = opt.random
            self.drop_last = True
        else:
            self.shuffle = False
            self.drop_last = False

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            # batch_size=1,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn=collate,
            drop_last=self.drop_last
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

def collate(batch):
    dim = 74000
    propensity = torch.cat([x['p'] for x in batch]).view(len(batch), 1)
    label = torch.cat([x['label'] for x in batch]).view(len(batch), 1)
    # propensity label batch * 1
    rows = []
    cols = []
    for i in range(len(batch)):
        cols.extend(batch[i]['idx'])
        for k in range(len(batch[i]['idx'])):
            rows.append(i)
    Rows = torch.from_numpy(np.array(rows, dtype='int64')).view(1, -1)
    Cols = torch.from_numpy(np.array(cols, dtype='int64')).view(1, -1)
    i = torch.cat((Rows, Cols), dim=0)
    vals = np.ones(len(cols), dtype='float32')
    v = torch.from_numpy(np.array(vals, dtype='float32'))
    batch_sparse_matrix = sparse.FloatTensor(i, v, torch.Size([len(batch), dim]))
    return {'p': propensity, 'feature': batch_sparse_matrix, 'label': label}