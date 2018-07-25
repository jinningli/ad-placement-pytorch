import crowdai
import argparse

parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--size', type=str, required=True)
args = parser.parse_args()

apikey = 'd2bb1449385c3a911f995b2b0f7dac1a'
challenge = crowdai.Challenge("CriteoAdPlacementNIPS2017", apikey)
scores = None
if args.size == 'small':
    scores = challenge.submit(args.data, small_test=True)
elif args.size == 'large':
    scores = challenge.submit(args.data, small_test=False)
print(scores)