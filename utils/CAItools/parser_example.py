#!/usr/bin/env python
from __future__ import print_function
import json

from utils.CAItools.criteo_dataset import CriteoDataset

train = CriteoDataset("../data/criteo_train_small.txt.gz")
"""
Arguments:
* `isTest` : Boolean
The `isTest` parameter is used to determine if its a test set (one which does not have cost/propensity information for every impression)
Hence in case of the training data, `isTest` should be `False`.

Ex:
test = CriteoDataset("../data/criteo_test_small.txt.gz", isTest=True)
"""

for _impression in train:
    print(_impression)
    """
        {
          "propensity": 336.294857951,
          "cost": 0.999,
          "id": "68965824",
          "candidates": [
            {
              0: 300,
              1: 600,
              2: 1,
              3: 1,
              4: 1,
              5: 1,
              6: 1,
              7: 1,
              8: 1,
              9: 1,
              10: 1,
              11: 1,
              12: 1,
              13: 1,
              14: 1,
              15: 1,
              16: 1,
              17: 1,
              18: 1,
              19: 1,
              20: 1
            },
            ...
            ...
            ...
          ]
        }
    """

train.close()
