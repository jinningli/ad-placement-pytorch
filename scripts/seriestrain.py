import os
import json
import codecs
from option.trainOption import TrainOptions
from data.cAIDataset import CAIDataset
from data.cAISparseDataset import CAISparseDataset
from dataLoader.lrDataLoader import LRDataLoader
from dataLoader.lrSparseDataLoader import LRSparseDataLoader
from model.lrModel import LRModel
from model.lrSparseModel import LRSparseModel
import utils.utils as util
from utils.logger import Logger
import time
import sys

def create_model(opt):
    model = None
    if opt.isTrain:
        if opt.sparse:
            model = LRSparseModel()
        else:
            model = LRModel()
    else:
        # from .test_model import TestModel
        # model = TestModel()
        pass
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    setattr(opt, 'isTrain', True)
    init_stdout = sys.stdout

    if opt.sparse:
        dataset = CAISparseDataset()
        lr_loader = LRSparseDataLoader()
    else:
        dataset = CAIDataset()
        lr_loader = LRDataLoader()

    dataset.initialize(opt)
    lr_loader.initialize(dataset=dataset, opt=opt)

    for c in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]:
        setattr(opt, 'propensity', 'min')
        setattr(opt, 'clip_value', c)
        setattr(opt, 'save_epoch_freq', 1)
        setattr(opt, 'name', 'clip_' + str(int(c)))
        print(opt)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)

        logfile = open(os.path.join(expr_dir, "log.txt"), 'a+')
        sys.stdout = Logger(init_stdout, logfile)

        model = create_model(opt)
        total_steps = 0

        for epoch in range(opt.epoch):
            epoch_iter = 0
            t_data = 0.0
            epoch_start_time = time.time()
            iter_start_time = time.time()

            for i, data in enumerate(lr_loader):
                # print('[' + str(epoch) + "][" + str(epoch_iter) + ']')
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    res = ''
                    res += '[' + str(epoch) + "][" + str(epoch_iter) + '/' + str(len(dataset)) + '] Loss: %.5f'%(model.get_current_losses()) + ' Time: %.2f'%(time.time() - iter_start_time)
                    print(res)
                    pass

                iter_start_time = time.time()

            print('[' + str(epoch) + '] End of epoch. Time: %.2f'%(time.time() - epoch_start_time))
            if epoch % opt.save_epoch_freq == 0 or epoch == opt.epoch - 1:
                model.save_networks('latest')
                model.save_networks(epoch)

            if not opt.lr_policy == 'same':
                model.update_learning_rate()

        logfile.close()
        sys.stdout = init_stdout

        model.cpu()
        model = None
