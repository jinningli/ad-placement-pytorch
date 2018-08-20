import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.sparse as sparse

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

def to_prediction_str(id, preds, opt):
    if opt.nopost:
        res = ['%d:%.8f' % (i, p) for (i, p) in enumerate(preds)]
    else:
        res = ['%d:%.2f' % (i, p) for (i, p) in enumerate(preds)]
    return '%d;%s' % (id, ','.join(res))

shift = 1.1875
scale = 850100
maxplus = 15

def post_process(pred, id, opt):
    if not opt.nopost:
        pred = shifted_scaled_sigmoid(pred, shift, scale)
        m = pred.argmax()
        pred[m] = pred[m] + maxplus
    pred_str = to_prediction_str(id, pred, opt)

    scores = pred
    best_score = np.max(scores)
    scores_with_offset = scores - best_score
    prob_scores = np.exp(scores_with_offset)
    # pred_str += '\n' + to_prediction_str(id, prob_scores / np.sum(prob_scores), opt)
    return pred_str + '\n'

def to_csr(idxs, vals, dim=74000):
    vals = np.ones(len(idxs), dtype='int64')  # TODO use value?
    rows = np.zeros((len(idxs)), dtype='int64')
    cols = np.array(idxs, dtype='int64')
    values = np.array(vals, dtype='float32')
    return sp.csr_matrix((values, (rows, cols)), shape=(1, dim))

def get_sparse_tensor(idxs, vals, dim=74000):
    rows = torch.from_numpy(np.zeros(len(idxs), dtype='int64')).view(1, -1)
    cols = torch.from_numpy(np.array(idxs, dtype='int64')).view(1, -1)
    i = torch.cat((rows, cols), dim=0)
    vals = np.ones(len(idxs), dtype='int64') #TODO use value?
    v = torch.from_numpy(np.array(vals, dtype='float32'))
    sparse_matrix = sparse.FloatTensor(i, v, torch.Size([1, dim]))
    # print(sparse_matrix)
    return sparse_matrix

def get_kd_sparse_tensor(idxsls, dim=74000):
    rowsls = []
    for k in range(len(idxsls)):
        for j in range(len(idxsls[k])):
            rowsls.append(k)
    rows = torch.from_numpy(np.array(rowsls, dtype='int64')).view(1, -1)
    idxseries = []
    for k in idxsls:
        idxseries.extend(k)
    cols = torch.from_numpy(np.array(idxseries, dtype='int64')).view(1, -1)
    i = torch.cat((rows, cols), dim=0)
    vals = np.ones(len(idxseries), dtype='int64') #TODO use value?
    v = torch.from_numpy(np.array(vals, dtype='float32'))
    sparse_matrix = sparse.FloatTensor(i, v, torch.Size([len(idxsls), dim]))
    # print(sparse_matrix)D
    return sparse_matrix