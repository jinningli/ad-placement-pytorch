from __future__ import print_function
import utils.CAItools.utils as utils
from itertools import (takewhile,repeat)
import numpy as np
import gzip

class CriteoPrediction:
    def __init__(self, filepath, isGzip=False, debug=False):
        self.debug = debug
        if filepath.endswith(".gz") or isGzip:
            self.fp = gzip.open(filepath, "rb")
        else:
            self.fp = open(filepath, "rb")
        self.count_max_instances()
        self.fp.close()
        self.fp = open(filepath, "r")

    def count_max_instances(self):
        bufgen = takewhile(lambda x: x, (self.fp.read(1024*1024) for _ in repeat(None)))
        self.max_instances = sum( buf.count(b'\n') for buf in bufgen )
        self.fp.seek(0)

    def __iter__(self):
        return self

    def parse_valid_line(self, line):
        line = line.strip()
        impression_id_marker = line.index(";")
        impression_id = line[:impression_id_marker]
        assert impression_id != ""

        prediction_string = line[impression_id_marker+1:]
        predictions = prediction_string.strip().split(",")
        predictions = [x.split(":") for x in predictions]

        #TODO: Add validation later
        parsed_predictions = np.zeros(len(predictions))
        for _pred in predictions:
            action = int(_pred[0])
            score = np.float64(_pred[1])

            #TODO: Add assertions and error handling later
            parsed_predictions[action] = score
        return {
                "id" : impression_id,
                "scores" : parsed_predictions
                }

    def next(self):
        try:
            line = next(self.fp)
            return self.parse_valid_line(line)
        except StopIteration:
            raise StopIteration

    def __next__(self):
        return self.next()

    def close(self):
        self.__del__()

    def __del__(self):
        self.fp.close()
