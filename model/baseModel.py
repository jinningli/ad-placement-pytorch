import os
import torch
from collections import OrderedDict
import torch.nn as nn

class BaseModel(nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu = opt.gpu
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.schedulers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    # save models to the disk
    def save_networks(self, which_epoch):
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         save_filename = 'Ep_%s_%s.save' % (which_epoch, name)
        #         save_path = os.path.join(self.save_dir, save_filename)
        #         net = getattr(self, 'net' + name)
        #
        #         if self.gpu >= 0 and torch.cuda.is_available():
        #             torch.save(net.module.cpu().state_dict(), save_path)
        #             net.cuda(self.gpu)
        #         else:
        #             torch.save(net.cpu().state_dict(), save_path)
        if self.opt.propensity != 'no':
            self.criterion = None
            self.loss = None
        save_filename = '%s_%s.save' % (which_epoch, self.name())
        save_path = os.path.join(self.save_dir, save_filename)
        if not self.name() == 'latest':
            print('Saving Model to ' + save_path)
        if self.gpu >= 0 and torch.cuda.is_available():
            torch.save(self.cpu().state_dict(), save_path)
            self.cuda(self.gpu)
        else:
            self.save(self.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, which_epoch):
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         save_filename = 'Ep_%s_%s.save' % (which_epoch, name)
        #         save_path = os.path.join(self.save_dir, save_filename)
        #         net = getattr(self, 'net' + name)
        #         if self.gpu >= 0 and torch.cuda.is_available():
        #             net.module.load_state_dict(torch.load(save_path))
        #         else:
        #             net.load_state_dict(torch.load(save_path))
        save_filename = '%s_%s.save' % (which_epoch, self.name())
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading Model from ' + save_path)

        pretrained_dict = torch.load(save_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        if self.gpu >= 0 and torch.cuda.is_available():
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(model_dict)

    # print network information
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')