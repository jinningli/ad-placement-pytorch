#!/usr/bin/env python
from __future__ import print_function

from utils.CAItools.criteo_dataset import CriteoDataset
from utils.CAItools.criteo_prediction import CriteoPrediction

import numpy as np
import utils.CAItools.utils as utils

def grade_predictions(predictions_path, gold_labels_path, expected_number_of_predictions=False, force_gzip=False, _context=False, salt_swap=False, inverse_propensity_in_gold_data=True, jobfactory_utils=False, _debug = False):
    gold_data = CriteoDataset(gold_labels_path, isGzip=force_gzip, inverse_propensity=inverse_propensity_in_gold_data)
    predictions = CriteoPrediction(predictions_path, isGzip=force_gzip)

    # Instantiate variables
    pos_label = 0.001
    neg_label = 0.999

    max_instances = predictions.max_instances
    if expected_number_of_predictions:
        if max_instances != expected_number_of_predictions:
            raise Exception("The prediction file is expected to contain predictions for {} impressions. But the submitted file contains predictins for {} impressions.".format(expected_number_of_predictions, max_instances))

    num_positive_instances = 0
    num_negative_instances = 0

    #NewPolicy - Stochastic
    prediction_stochastic_numerator = np.zeros(max_instances, dtype = np.float)
    prediction_stochastic_denominator = np.zeros(max_instances, dtype = np.float)

    impression_counter = 0
    for _idx, _impression in enumerate(gold_data):
        # TODO: Add Validation
        prediction = next(predictions)
        if _impression['id'] != prediction['id']:
            raise Exception("`prediction_id` {} doesnot match the corresponding `impression_id`. Please ensure that the lines in the prediction file are in the same order as in the test set.".format(prediction['id']))

        scores = prediction["scores"]

        label = _impression["cost"]
        propensity = _impression["propensity"]
        num_canidadates = len(_impression["candidates"])

        rectified_label = 0

        if label == pos_label:
            rectified_label = 1
            num_positive_instances += 1
        elif label == neg_label:
            num_negative_instances += 1
        else:
            raise Exception("Unknown cost label for impression_id {}".format(_impression['id']))

        if label == pos_label:
            log_weight = 1.0
        elif label == neg_label:
            log_weight = 10.0
        else:
            raise Exception("Unknown cost label for impression_id {}".format(_impression['id']))

        #For deterministic policy
        best_score = np.max(scores)
        best_classes = np.argwhere(scores == best_score).flatten()

        #For stochastic policy
        score_logged_action = None
        score_normalizer = 0.0

        scores_with_offset = scores - best_score
        prob_scores = np.exp(scores_with_offset)

        score_normalizer = np.sum(prob_scores)

        logged_action_index = 0
        if salt_swap:
            logged_action_index = utils.compute_integral_hash(_impression['id'], salt_swap, len(_impression["candidates"]))

        score_logged_action = prob_scores[logged_action_index]

        prediction_stochastic_weight = 1.0 * score_logged_action / (score_normalizer * propensity)
        prediction_stochastic_numerator[_idx] = rectified_label * prediction_stochastic_weight
        prediction_stochastic_denominator[_idx] = prediction_stochastic_weight

        impression_counter += 1 #Adding this as _idx is not available out of this scope
        if _idx % 100 == 0:
            if _context and jobfactory_utils:
                jobfactory_utils.update_progress(_context, _idx*100.0/max_instances)
            if _debug: print('.', end='')

    gold_data.close()
    predictions.close()

    modified_denominator = num_positive_instances + 10*num_negative_instances
    scaleFactor = np.sqrt(max_instances) / modified_denominator

    if _debug:
        print('')
        print("Num[Pos/Neg]Test Instances:", num_positive_instances, num_negative_instances)
        print("MaxID; curId", max_instances, impression_counter)
        print("Approach & IPS(*10^4) & StdErr(IPS)*10^4 & SN-IPS(*10^4) & StdErr(SN-IPS)*10^4 & AvgImpWt & StdErr(AvgImpWt) \\")

    def compute_result(approach, numerator, denominator):
        # numerator[k] = rectified_label * 1.0 * score_logged_action / (score_normalizer * propensity)
        # = rectified_label * 1.0 * prob_scores[logged_action_index] / (np.sum(prob_scores) * propensity)

        # modified_denominator = num_positive_instances + 10*num_negative_instances

        IPS = numerator.sum(dtype = np.longdouble) / modified_denominator

        # scaleFactor = np.sqrt(max_instances) / modified_denominator
        IPS_std = 2.58 * numerator.std(dtype = np.longdouble) * scaleFactor        #99% CI
        ImpWt = denominator.sum(dtype = np.longdouble) / modified_denominator
        ImpWt_std = 2.58 * denominator.std(dtype = np.longdouble) * scaleFactor    #99% CI
        SNIPS = IPS / ImpWt

        normalizer = ImpWt * modified_denominator

        #See Art Owen, Monte Carlo, Chapter 9, Section 9.2, Page 9
        #Delta Method to compute an approximate CI for SN-IPS
        Var = np.sum(np.square(numerator) +\
                        np.square(denominator) * SNIPS * SNIPS -\
                        2 * SNIPS * np.multiply(numerator, denominator), dtype = np.longdouble) / (normalizer * normalizer)

        SNIPS_std = 2.58 * np.sqrt(Var) / np.sqrt(max_instances)                 #99% CI

        _response = {}
        _response["ips"] = IPS*1e4
        _response["ips_std"] = IPS_std*1e4
        _response["snips"] = SNIPS*1e4
        _response["snips_std"] = SNIPS_std*1e4
        _response["impwt"] = ImpWt
        _response["impwt_std"] = ImpWt_std
        _response["max_instances"] = max_instances

        return _response

    return compute_result('NewPolicy-Stochastic', prediction_stochastic_numerator, prediction_stochastic_denominator)

if __name__ == "__main__":
    gold_labels_path = "data/cntk_train_small.txt.gz"
    predictions_path = "data/predictions.txt.gz"
    print(grade_predictions(predictions_path, gold_labels_path, _debug=True))
