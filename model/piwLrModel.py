from model.baseModel import BaseModel
import torch
import model.networks as networks
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model.networks import init_net
from model.networks import DynamicEmbedding
import numpy as np
import math
from torch.nn.parameter import Parameter

class PiwLRModel(BaseModel):
    def name(self):
        return 'PiwLRModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['loss']
        # self.info_names = ['weight']
        self.model_names = ['fc1', 'fc2']
        self.fc1 = init_net(nn.Linear(opt.max_idx, 4096), init_type='normal', gpu=self.opt.gpu)
        self.fc2 = init_net(nn.Linear(4096, 1), init_type='normal', gpu=self.opt.gpu)

        if opt.isTrain:
            if opt.propensity == 'no':
                self.criterion = nn.BCELoss(size_average=True)
                if opt.gpu >= 0:
                    self.criterion.cuda(opt.gpu)
            else:
                self.criterion = None

            self.schedulers = []
            self.optimizers = []
            self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=1e-3)
            self.optimizers.append(self.optimizer)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks()

    def set_input(self, input):
        propensity = None
        label = None
        id = input['id']
        self.index = input['index']

        if self.opt.isTrain:
            propensity = input['propensity'].view(self.opt.batchSize, 1)
            label = input['label']

        feature = input['feature']

        if self.gpu >= 0:
            if self.opt.isTrain:
                propensity = propensity.cuda(self.gpu, async=True)
                label = label.cuda(self.gpu, async=True)
            feature = feature.cuda(self.gpu, async=True)

        if self.opt.isTrain:
            self.propensity = propensity
            self.label = label

        self.id = id
        self.feature = feature

    def forward(self):
        if self.opt.isTrain:
            self.label = Variable(self.label)

        self.feature = Variable(self.feature)

        out = self.fc1(self.feature)
        out = F.relu(out)
        out = self.fc2(out)
        # out = F.sigmoid(out)
        self.pred = out

    # no backprop gradients
    def test(self):
        self.forward()
        return {'ids': self.id.cpu().numpy().reshape(-1), 'preds': self.pred.data.cpu().numpy().reshape(-1)}

    def _test(self, input):
        self.set_input(input)
        self.forward()
        return {'ids': input['id'], 'preds': self.pred.data.cpu().numpy()}

    def backward(self):
        if self.opt.propensity == 'no':
            self.loss = self.criterion(self.pred, self.label)
        elif self.opt.propensity == 'naive':
            self.criterion = nn.BCELoss(weight=self.propensity, size_average=True)
            self.loss = self.criterion(self.pred, self.label)
            if self.opt.gpu >= 0:
                self.loss.cuda(self.opt.gpu)
        elif self.opt.propensity == 'min':
            cval = self.opt.clip_value
            self.propensity[self.propensity >= cval] = cval
            self.criterion = nn.BCELoss(weight=self.propensity, size_average=True)
            self.loss = self.criterion(self.pred, self.label)
            if self.opt.gpu >= 0:
                self.loss.cuda(self.opt.gpu)
        elif self.opt.propensity == 'piw':
            pred = self.pred.data
            piw = []
            for k in range(self.opt.batchSize):
                arr = Variable(pred[self.index[k]:self.index[k+1]])
                piw.append(float(F.softmax(arr - torch.max(arr), dim=0)[0].data))
            self.piw = torch.from_numpy(np.array(piw, dtype='float32')).view(self.opt.batchSize, 1).cuda(self.gpu, async=True)
            self.weight = self.piw * self.propensity
            self.criterion = nn.BCEWithLogitsLoss(weight=self.weight, size_average=True)
            self.prediction = torch.cat([self.pred[self.index[k]] for k in range(self.opt.batchSize)], dim=0).view(self.opt.batchSize, 1)
            self.loss = self.criterion(self.prediction, self.label)
            if self.opt.gpu >= 0:
                self.loss.cuda(self.opt.gpu)
        else:
            raise NotImplementedError('No such propensity mode')
        # print(float(self.loss.data.cpu()))
        self.loss.backward()
        # print(self.loss.data)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return float(self.loss.data.cpu().numpy())
