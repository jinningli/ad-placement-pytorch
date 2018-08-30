import os
from option.testOption import TestOptions
from data.cAITestDataset import CAITestDataset
from model.lrModel import LRModel
from utils.utils import post_process
from dataLoader.lrDataLoader import LRDataLoader
from itertools import groupby
from model.piwLrModel import PiwLRModel
from model.lrSparseModel import LRSparseModel
from model.POEM import POEMModel
import numpy as np
import time

def create_model(opt):
    model = None
    if opt.sparse:
        model = LRSparseModel()
    elif opt.propensity == 'piw' or opt.propensity == 'piwMSE':
        model = PiwLRModel()
    elif opt.propensity == 'POEM':
        model = POEMModel()
    else:
        model = LRModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

if __name__ == '__main__':
    opt = TestOptions().parse()
    setattr(opt, 'isTrain', False)
    # opt.nThreads = 1
    # opt.batchSize = 1

    dataset = CAITestDataset()
    dataset.initialize(opt)

    lr_loader = LRDataLoader()
    lr_loader.initialize(dataset=dataset, opt=opt)

    model = create_model(opt)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(model.save_dir):
        os.mkdir(model.save_dir)
    if not os.path.exists(os.path.join(model.save_dir, "test")):
        os.mkdir(os.path.join(model.save_dir, "test"))

    cnt = 0
    print('Start Predicting')
    timestr = time.strftime("_%m%d%H%M%S", time.localtime())
    with open(os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt'), 'w+') as output:
        ls = []
        epoch_start_time = time.time()
        iter_start_time = time.time()

        for i, data in enumerate(lr_loader): # for every batch
            model.set_input(data)
            res = model.test()
            preds = res['preds'].tolist()
            ids = res['ids'].tolist()
            cnt += len(preds)
            if cnt % opt.display_freq == 0:
                print('[' + str(cnt) + '/' + str(len(dataset)) + ']' + ' Time: %.2f'%(time.time() - iter_start_time))
            for k in range(len(preds)):
                ls.append((ids[k], preds[k]))
            iter_start_time = time.time()

        groups = groupby(ls, key=lambda x: x[0])
        for id, group in groups:
            arrayls = []
            for item in group:
                arrayls.append(item[1])
            output.write(post_process(pred=np.array(arrayls, dtype='float32'), id=id, opt=opt))

    os.system('gzip ' + os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt'))

    # Detach the memory
    save_dir = model.save_dir
    model.cpu()
    model = None
    dataset = None
    lr_loader = None

    print('Prediction Saved in ' + os.path.join(save_dir, "test", 'pred' + timestr + '.txt.gz'))
    print('>>Submit Now: (large/small/no)?')
    s = input()
    if s == 'no':
        exit()
    os.system('python3 submit.py --data ' + os.path.join(save_dir, "test", 'pred' + timestr + '.txt.gz') + ' --size ' + s)
    print('python3 submit.py --data ' + os.path.join(save_dir, "test", 'pred' + timestr + '.txt.gz') + ' --size ' + s)



