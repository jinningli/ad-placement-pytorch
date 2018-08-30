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

from .networks import BCEWithLogitsLoss as BCELLoss

class POEMModel(BaseModel):
    def name(self):
        return 'POEMModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['loss']
        self.info_names = []
        self.model_names = ['fc1', 'fc2']
        self.fc1 = init_net(nn.Linear(opt.max_idx, 4096), init_type='normal', gpu=self.opt.gpu)
        self.fc2 = init_net(nn.Linear(4096, 1), init_type='normal', gpu=self.opt.gpu)

        class POEMLoss(nn.Module):
            def __init__(self):
                super(POEMLoss, self).__init__()

            def forward(self, weight, theta):
                return torch.sum(theta * weight) / weight.shape[0]

        if opt.isTrain:
            self.criterion = POEMLoss()

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

        if self.opt.isTrain:
            propensity = input['propensity'].view(self.opt.batchSize, 1)
            label = input['label']
            self.index = input['index']

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

        if self.opt.isTrain and (self.opt.piw_gradient or self.opt.propensity == 'piwMSE'):
            self.propensity = Variable(self.propensity)

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
        self.piw = None
        for k in range(self.opt.batchSize):
            arr = self.pred[self.index[k]:self.index[k + 1]]
            if k == 0:
                self.piw = torch.max(F.softmax(arr - torch.max(arr), dim=0))
            else:
                self.piw = torch.cat([self.piw, torch.max(F.softmax(arr - torch.max(arr), dim=0))], dim=0)
        self.weight = self.piw.view(self.opt.batchSize, 1) * self.propensity # weight here: Variable
        self.loss = self.criterion(self.weight, self.label)
        if self.opt.gpu >= 0:
            self.loss.cuda(self.opt.gpu)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return float(self.loss.data.cpu().numpy())
