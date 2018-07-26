import os
from option.testOption import TestOptions
from data.cAITestDataset import CAITestDataset
from model.lrModel import LRModel
from utils.utils import post_process
from dataLoader.lrDataLoader import LRDataLoader
from itertools import groupby
import numpy as np
import time

def create_model(opt):
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
        for i, data in enumerate(lr_loader): # for every batch
            model.set_input(data)
            res = model.test()
            preds = res['preds'].tolist()
            ids = res['ids'].tolist()
            cnt += len(preds)
            if cnt % opt.display_freq == 0:
                print('[' + str(cnt) + '/' + str(len(dataset)) + ']')
            for k in range(len(preds)):
                ls.append((ids[k], preds[k]))
        groups = groupby(ls, key=lambda x: x[0])
        for id, group in groups:
            arrayls = []
            for item in group:
                arrayls.append(item[1])
            output.write(post_process(pred=np.array(arrayls, dtype='float32'), id=id))
    os.system('gzip ' + os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt'))
    print('Prediction Saved in ' + os.path.join(model.save_dir, "test", 'pred' + timestr + '.txt.gz'))
    print('>>Submit Now: (large/small/no)?')
    s = input()
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



