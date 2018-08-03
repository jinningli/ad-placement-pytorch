import os
from option.testOption import TestOptions
from data.cAITestDataset import CAITestDataset
from model.lrModel import LRModel
from utils.utils import post_process
from dataLoader.lrDataLoader import LRDataLoader
from itertools import groupby
import numpy as np
import time
from utils.logger import Logger
import sys
import torch.nn as nn
import torch
import json

def create_model(opt):
    model = LRModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

if __name__ == '__main__':
    opt = TestOptions().parse()
    setattr(opt, 'isTrain', False)
    init_stdout = sys.stdout

    dataset = CAITestDataset()
    dataset.initialize(opt)

    lr_loader = LRDataLoader()
    lr_loader.initialize(dataset=dataset, opt=opt)

    testres = []

    for c in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]:

        setattr(opt, 'name', 'clip_' + str(int(c)))
        print(opt)

        model = create_model(opt)

        # if torch.cuda.device_count() > 1: #############################################
        #     model = nn.DataParallel(model)
        #
        # model.to([1, 2, 3])

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists(model.save_dir):
            os.mkdir(model.save_dir)
        if not os.path.exists(os.path.join(model.save_dir, "test")):
            os.mkdir(os.path.join(model.save_dir, "test"))

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        logfile = open(os.path.join(expr_dir, "testlog.txt"), 'w+')
        sys.stdout = Logger(init_stdout, logfile)

        cnt = 0
        print('Start Predicting')
        timestr = time.strftime("_%m%d%H%M%S", time.localtime())
        with open(os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt'), 'w+') as output:
            ls = []
            epoch_start_time = time.time()
            iter_start_time = time.time()

            for i, data in enumerate(lr_loader): # for every batch
                # data_paral = data.to([1, 2, 3]) ######################################
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
                output.write(post_process(pred=np.array(arrayls, dtype='float32'), id=id))

        os.system('gzip ' + os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt'))
        print('Prediction Saved in ' + os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt.gz'))
        print('>>Submit Now: (large/small/no)?')
        s = 'small'
        if s == 'no':
            exit()
        import crowdai
        apikey = 'd2bb1449385c3a911f995b2b0f7dac1a'
        challenge = crowdai.Challenge("CriteoAdPlacementNIPS2017", apikey)
        scores = None
        if s == 'small':
            scores = challenge.submit(os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt.gz'), small_test=True)
        elif s == 'large':
            scores = challenge.submit(os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt.gz'), small_test=False)
        print(scores)

        testres.append(scores)

        logfile.close()
        sys.stdout = init_stdout

        model.cpu()
        model = None

    with open('testlog.txt', 'w+') as output:
        output.write(json.dumps(testres))



