# ad-placement-pytorch
Completed Logistic Regression. Other methods and applying of propensity are to be completed.

Get a highest score **IPS=59.249** of late submission in <a href='https://www.crowdai.org/topics/solution-sharing/discussion'>Criteo Ad Placement Challenge</a>.

#### Get the code:
```
git clone https://github.com/jinningli/ad-placement-pytorch.git
```

#### Requirements:
```
CUDA
pytorch
scipy
numpy
crowdai
```

#### Build dataset
```
Download from https://www.crowdai.org/challenges/nips-17-workshop-criteo-ad-placement-challenge/dataset_files
gunzip criteo_train.txt.gz
gunzip criteo_test_release.txt.gz
mkdir datasets/crowdai
mv criteo_train.txt datasets/crowdai/criteo_train.txt
mv criteo_test_release.txt datasets/crowdai/test.txt
cd datasets/crowdai
python3 getFirstLine.py --data criteo_train.txt --result train.txt
```

#### Train

```
python3 train.py --dataroot datasets/criteo --name Saved_Name --batchSize 5000 --gpu 0 --no_cache
```

#### Test

```
python3 test.py --dataroot datasets/criteo --name Saved_Name --batchSize 5000 --gpu 0 --no_cache
```

Checkpoints and Test result will be saved in `checkpoints/Saved_Name`.

#### All options

`--display_freq` Frequency of showing train/test results on screen.

`--continue_train` Continue training.

`--lr_policy` Learning rate policy: same|lambda|step|plateau.

`--epoch` How many epochs.

`--save_epoch_freq` Epoch frequency of saving model.

`--gpu` Which gpu device, -1 for CPU.

`--dataroot` Dataroot path.

`--checkpoints_dir` Models are saved here.

`--name` Name for saved directory.

`--batchSize` Batch size.

`--lr` Learning rate.

`--which_epoch` Which epoch to load? default is the latest.

`--no_cache` Don't save processed dataset for faster loading next time.

`--random` Randomize (Shuffle) input data.

`--nThreads` Number of threads for loading data.
