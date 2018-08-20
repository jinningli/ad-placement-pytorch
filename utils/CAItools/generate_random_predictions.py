#!/usr/bin/env python
from __future__ import print_function
from utils.CAItools.criteo_dataset import CriteoDataset
import numpy as np
import gzip
import argparse

parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--test_set', dest='test_set', action='store', required=True)
parser.add_argument('--output_path', dest='output_path', action='store', required=True)
args = parser.parse_args()

print("Reading test set from : ",args.test_set)
data = CriteoDataset(args.test_set, isTest=True)
"""
The `isTest` parameter is used to determine if its a test set (one which does not have cost/propensity information for every impression)
Hence in case of the training data, `isTest` should be `False`.
"""

output = gzip.open(args.output_path, "wb")

def _format_predictions(predictions):
    predictions = ["{}:{}".format(idx, p) for idx, p in enumerate(predictions)]
    predictionline = "{};{}".format(_impression["id"], ",".join(predictions))
    return predictionline

def _policy(candidates):
    num_of_candidates = len(candidates)
    predictions = np.random.rand(num_of_candidates)*10
    return predictions

for _idx, _impression in enumerate(data):
    predictions = _policy(_impression["candidates"])
    predictionline = _format_predictions(predictions)
    predictionline += "\n"
    predictionline = predictionline.encode() #Note this is important for python3 compatibility as we are writing in "wb" mode
    output.write(predictionline)

    if _idx % 500 == 0:
        print("Processed {} impressions...".format(_idx))

output.close()
print("Successfully Wrote predictions file to : ",args.output_path)
