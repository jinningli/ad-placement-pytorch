import os
import codecs
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, required=True, help='Path of input data file (whole impression)')
parser.add_argument('--rate', type=float, default=0.75, help='Split rate')
args = parser.parse_args()

with codecs.open(args.data, 'r+') as input:
    cnt = 0
    ftrain = open('_train.txt', 'w+')
    fvalid = open('_valid.txt', 'w+')
    isvalid = None
    for line in input:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
        split = line.split('|')
        if len(split) == 4:
            isvalid = random.random() > args.rate
            if isvalid:
                fvalid.write(line)
            else:
                ftrain.write(line)
            continue

        if len(split) == 2:
            if isvalid:
                fvalid.write(line)
            else:
                ftrain.write(line)


