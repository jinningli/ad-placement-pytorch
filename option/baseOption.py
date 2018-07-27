import argparse
import os
import torch
import utils.utils as util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # important
        self.parser.add_argument('--gpu', type=int, default=0, help='which gpu device, -1 for CPU')
        self.parser.add_argument('--dataroot', required=True, help='dataroot path')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='Saved_Experiment', help='Name for saved directory')
        self.parser.add_argument('--batchSize', type=int, default=32, help='batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--no_cache', action='store_true', help='dont store the preprocessed dataset in a pickle cache')
        # ignorable
        self.parser.add_argument('--size_idx', type=int, default=200, help='Set max length of sentence')
        self.parser.add_argument('--max_idx', type=int, default=74000, help='Set max length of sentence')
        self.parser.add_argument('--random', type=bool, default=True, help='randomize input data')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # set gpu
        torch.cuda.set_device(opt.gpu)

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        self.opt = opt
        return self.opt
