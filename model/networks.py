from torch.optim import lr_scheduler
from torch.nn import init
import torch
import torch.nn as nn

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch - 100) / float(100 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_net(net, init_type='normal', gpu=-1):
    if gpu >= 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu)
        net = torch.nn.DataParallel(net, [gpu])
    init_weights(net, init_type)
    return net

def init_weights(net, init_type='normal', gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)


class DynamicEmbedding(nn.Module):
    def __init__(self, max_idx, embedding_dim=4, dropout_rate=None, method='avg'):
        """
        max_idx: max_idx of feature (200)
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DynamicEmbedding, self).__init__()

        assert method in ['avg', 'sum']

        self.max_idx = max_idx
        self.embedding_size = embedding_dim
        self.dropout_rate = dropout_rate
        self.method = method

        # initial layer
        self.embedding = nn.Embedding(max_idx, self.embedding_size, padding_idx=0)

        self.is_dropout = False
        if self.dropout_rate is not None:
            self.is_dropout = True
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, input):
        """
        input: relative id 
        dynamic_ids: Batch_size * Max_feature_size(self define)  [[1, 2, 3, 4, 0, 0],[5, 6, 7, 0, 0, 0]] 
        dynamic_lengths: Batch_size * 1                          [[4], [3]]
        return: Batch_size * Embedding_size
        """
        # B*M
        dynamic_ids_tensor = input[0]
        dynamic_length_tensor = input[1]
        # if self.use_cuda:
        #     dynamic_ids_tensor = dynamic_ids_tensor.cuda()
        #     dynamic_length_tensor = dynamic_length_tensor.cuda()

        batch_size = dynamic_ids_tensor.size()[0]
        # max_feature_size = dynamic_ids_tensor.size()[-1]

        # embedding layer B*M*E
        dynamic_embeddings_tensor = self.embedding(dynamic_ids_tensor)

        # dropout
        if self.is_dropout:
            dynamic_embeddings_tensor = self.dropout(dynamic_embeddings_tensor)

        # average B*M*E --AVG--> B*E
        dynamic_embedding = torch.sum(dynamic_embeddings_tensor, 1)

        if self.method == 'avg':
            # B*E
            # dynamic_lengths_tensor = dynamic_lengths_tensor.view(-1, 1).expand_as(dynamic_embedding)
            dynamic_embedding = dynamic_embedding / dynamic_length_tensor
        res = dynamic_embedding.view(batch_size, self.embedding_size)
        # B*E
        return res
