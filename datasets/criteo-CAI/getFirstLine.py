import os
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, required=True, help='Path of input data file')
parser.add_argument('--result', type=str, default=None, help='Path of output cleaned data')
args = parser.parse_args()

if args.result == None:
    setattr(args, 'result', args.data.replace('.txt', '_clean.txt'))

print(args)

cnt = 0

with codecs.open(args.data, 'r+', 'utf-8') as Input:
    with codecs.open(args.result, 'w+', 'utf-8') as Output:
        for line in Input:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            if line.find('|l') != -1:
                Output.write(line)




