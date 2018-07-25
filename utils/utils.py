import os
import numpy as np

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shifted_scaled_sigmoid(x, shift=0.0, scale=1.0):
    s = 1 / (1 + np.exp(-x + shift))
    return (s * scale).round(2)

def to_prediction_str(id, preds):
    res = ['%d:%0.2f' % (i, p) for (i, p) in enumerate(preds)]
    return '%d;%s' % (id, ','.join(res))

shift = 1.1875
scale = 850100
maxplus = 15

def post_process(pred, id):
    pred = shifted_scaled_sigmoid(pred, shift, scale)
    m = pred.argmax()
    pred[m] = pred[m] + maxplus
    pred_str = to_prediction_str(id, pred)
    return pred_str + '\n'
