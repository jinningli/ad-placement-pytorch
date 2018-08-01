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

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ss = torch.mm(input, self.weight.t())
        return torch.add(ss, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class LRSparseModel(BaseModel):
    def name(self):
        return 'LRSparse'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['loss']
        self.model_names = ['fc1', 'fc2']
        self.fc1 = init_net(MyLinear(opt.max_idx, 4096), init_type='normal', gpu=self.opt.gpu)
        self.fc2 = init_net(MyLinear(4096, 1), init_type='normal', gpu=self.opt.gpu)

        if opt.isTrain:
            if opt.propensity == 'no':
                self.criterion = nn.BCELoss(size_average=True)
                if opt.gpu >= 0:
                    self.criterion.cuda(opt.gpu)
            else:
                self.criterion = None

            self.schedulers = []
            self.optimizers = []
            self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=1e-5)
            # self.optimizer = torch.optim.SparseAdam(self.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks()

    def set_input(self, input):
        propensity = None
        label = None

        if self.opt.isTrain:
            # propensity = input['p'].view(self.opt.batchSize, 1) #######################################
            label = input['label']
        else:
            self.id = input['id']

        feature = input['feature']

        if self.gpu >= 0:
            if self.opt.isTrain:
                # propensity = propensity.cuda(self.gpu, async=True)#################################
                label = label.cuda(self.gpu, async=True)
            feature = feature.cuda(self.gpu, async=True)

        if self.opt.isTrain:
            # self.propensity = propensity################################################
            self.label = label
        self.feature = feature

    def forward(self):
        if self.opt.isTrain:
            # self.propensity = Variable(self.propensity)#############################################
            self.label = Variable(self.label)

        self.feature = Variable(self.feature)

        out = self.fc1(self.feature)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
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
