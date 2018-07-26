from dataLoader.baseDataLoader import BaseDataLoader
from torch.utils.data import DataLoader
import torch

class LRDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDataLoader'

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
            num_workers=int(opt.nThreads),
            # collate_fn=collate
            drop_last=self.drop_last
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

# def collate(batch):
#     id = [x['id'] for x in batch]
#     propensity = torch.cat([x['p'] for x in batch])
#     f0 = torch.cat([x['f0'] for x in batch])
#     f1 = torch.cat([x['f1'] for x in batch])
#     val = torch.cat([x['value'] for x in batch if x['value'] is not None])
#     label = torch.cat([x['label'] for x in batch])
#     idx = torch.cat([x['idx'] for x in batch if x['idx'] is not None])
#
#     return {'id': id, 'p': propensity, 'f0': f0, 'f1': f1,
#                                       'idx': idx, 'value': val, 'label': label}