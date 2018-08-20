# Remember to generate the random predictions by running inside the repository
# python generate_random_prediction.py > data/predictions.txt
from __future__ import print_function
import crowdai
import argparse

parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--api_key', dest='api_key', action='store', required=True)
parser.add_argument('--predictions', dest='predictions', action='store', required=True)
args = parser.parse_args()

challenge = crowdai.Challenge("CriteoAdPlacementNIPS2017", args.api_key)

scores = challenge.submit(args.predictions, small_test=False)

"""
NOTE: In case of predictions for the actual test set, please pass `small_Test=False`

scores = challenge.submit(args.predictions, small_test=False)
"""

print(scores)

"""
{
  "impwt_std": 0.00064745289554913,
  "ips_std": 2.6075584296658,
  "snips": 6603.0581686235,
  "max_instances": 4027,
  "ips": 24.30130041425,
  "impwt": 0.0036803099099937,
  "message": "",
  "snips_std": 17.529346134878,
  "job_state": "CrowdAI.Event.Job.COMPLETE"
}
"""
