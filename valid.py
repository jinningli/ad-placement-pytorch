import os
from option.testOption import TestOptions
from data.cAITestDataset import CAITestDataset
from model.lrModel import LRModel
from utils.utils import post_process
from dataLoader.lrDataLoader import LRDataLoader
from itertools import groupby
import numpy as np
import time
from utils.CAItools.compute_score import grade_predictions

def create_model(opt):
    model = LRModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

def assertDataroot(opt):
    with open(os.path.join('checkpoints', opt.name, 'opt.txt'), 'r+') as input:
        for line in input:
            if line.find('dataroot: ') != -1:
                if not line.replace('dataroot: ', '').replace('\n', '') == opt.dataroot:
                    print(line.replace('dataroot: ', ''))
                    print(opt.dataroot)
                    raise AssertionError('Dataroot not consistent')

if __name__ == '__main__':
    opt = TestOptions().parse()
    setattr(opt, 'isTrain', False)
    setattr(opt, 'phase', 'valid')
    setattr(opt, 'split', True)
    # opt.nThreads = 1
    # opt.batchSize = 1
    assertDataroot(opt)

    dataset = CAITestDataset()
    dataset.initialize(opt)

    lr_loader = LRDataLoader()
    lr_loader.initialize(dataset=dataset, opt=opt)

    model = create_model(opt)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(model.save_dir):
        os.mkdir(model.save_dir)
    if not os.path.exists(os.path.join(model.save_dir, "val")):
        os.mkdir(os.path.join(model.save_dir, "val"))

    cnt = 0
    print('Start Predicting')
    timestr = time.strftime("_%m%d%H%M%S", time.localtime())
    with open(os.path.join(model.save_dir, "val", 'pred' + timestr + '.txt'), 'w+') as output:
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

    # os.system('gzip ' + os.path.join(model.save_dir, "val", 'pred' + timestr + '.txt'))

    # Detach the memory
    save_dir = model.save_dir
    model.cpu()
    model = None
    dataset = None
    lr_loader = None

    print('Prediction Saved in ' + os.path.join(save_dir, "val", 'pred' + timestr + '.txt'))

    print('Start Validation..')
    assert(os.path.exists(os.path.join(opt.dataroot, '_valid.txt')))
    gold_labels_path = os.path.join(opt.dataroot, '_valid.txt')
    predictions_path = os.path.join(save_dir, "val", 'pred' + timestr + '.txt')
    print(grade_predictions(predictions_path, gold_labels_path, _debug=True))




