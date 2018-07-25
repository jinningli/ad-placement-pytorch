import os
import json
import codecs
from option.trainOption import TrainOptions
from data.cAIDataset import CAIDataset
from dataLoader.lrDataLoader import LRDataLoader
from model.lrModel import LRModel

def create_model(opt):
    if opt.isTrain:
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

    dataset = CAIDataset()
    dataset.initialize(opt)

    lr_loader = LRDataLoader()
    lr_loader.initialize(dataset=dataset, opt=opt)

    model = create_model(opt)
    total_steps = 0

    for epoch in range(opt.epoch):
        epoch_iter = 0
        t_data = 0.0

        for i, data in enumerate(lr_loader):
            # print('[' + str(epoch) + "][" + str(epoch_iter) + ']')
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                res = ''
                res += '[' + str(epoch) + "][" + str(epoch_iter) + '] Loss: ' + str(model.get_current_losses())
                # for k, v in losses.items():
                #     res += '%s: %.3f ' % (k, v)
                # res += "| AvgTime: %.3f" % t_data
                print(res)
                pass

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        if not opt.lr_policy == 'same':
            model.update_learning_rate()
