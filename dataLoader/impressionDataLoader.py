from dataLoader.baseDataLoader import BaseDataLoader
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.sparse as sparse

class ImpressionDataLoader(BaseDataLoader):
    def name(self):
        return 'ImpressionDataLoader'

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
    propensity = torch.cat([x['propensity'] for x in batch]).view(len(batch), 1)
    label = torch.cat([x['label'] for x in batch]).view(len(batch), 1)
    id = torch.cat([x['id'] for x in batch]).view(len(batch), 1)
    # propensity, label, id: batch * 1
    feature = torch.cat([x['feature'] for x in batch], dim=0)

    lens = [x['length'] for x in batch]
    lens.insert(0, 0)
    lens = np.array(lens, dtype='int64')
    index = lens.cumsum()
    return {'id': id, 'propensity': propensity, 'label': label, 'feature': feature, 'index': index}