import os
from option.testOption import TestOptions
from data.cAITestDataset import CAITestDataset
from model.lrModel import LRModel
from utils.utils import post_process

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

    # lr_loader = LRDataLoader()
    # lr_loader.initialize(dataset=dataset, opt=opt)

    model = create_model(opt)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(model.save_dir):
        os.mkdir(model.save_dir)
    if not os.path.exists(os.path.join(model.save_dir, "test")):
        os.mkdir(os.path.join(model.save_dir, "test"))

    cnt = 0
    print('Start Predicting')
    with open(os.path.join(model.save_dir, "test", 'pred.txt'), 'w+') as output:
        for i, data in enumerate(dataset):
            cnt += 1
            if cnt % opt.display_freq == 0:
                print('[' + str(cnt) + '/' + str(len(dataset)) + ']')
            model.set_input(data)
            pred = model.test().reshape(-1)
            output.write(post_process(pred=pred, id=data['id']))
    os.system('gzip ' + os.path.join(model.save_dir, "test", 'pred.txt'))
    print('Prediction Saved in ' + os.path.join(model.save_dir, "test", 'pred.txt.gz'))
    print('>>Submit Now: (large/small/no)?')
    s = input()
    if s == 'no':
        exit()
    import crowdai
    apikey = 'd2bb1449385c3a911f995b2b0f7dac1a'
    challenge = crowdai.Challenge("CriteoAdPlacementNIPS2017", apikey)
    scores = None
    if s == 'small':
        scores = challenge.submit(os.path.join(model.save_dir, "test", 'pred.txt.gz'), small_test=True)
    elif s == 'large':
        scores = challenge.submit(os.path.join(model.save_dir, "test", 'pred.txt.gz'), small_test=False)
    print(scores)



