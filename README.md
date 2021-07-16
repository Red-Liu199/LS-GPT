# LS-GPT
## Requirements
* python=3.6
* pytorch=1.2.0
* trasformers=3.5.1
## Dataset
Before training and evaluation, you need to extract the dataset from zip files:
```
unzip data.zip
unzip db.zip
unzip extra_data.zip
unzip extra_db.zip
```
## Training
* Note: All the `.sh` files contain comments. Read them carefully and you can change the parameters such as `cuda_device` and `spv_proportion` to satisfy your own need.
### Supervised training over all data in MultiWOZ:
```
bash train.sh
```
### Semi-supervised training with partial labeled data in MultiWOZ

If you haven't pretrained the model on labeled data:
```
bash pretrain.sh
```
Then you can conduct self training experiment:
```
bash train_ST.sh
```
or semi variational learning experiment:
```
bash train_VL.sh
```
Note that Semi-VL needs two pretrained models including generative model and inference model. The original script `pretrain.sh` only pretrains the generative model. Set the parameter `posterior` in `pretrain.sh` to True to pretrain the inference model.
### Semi-supervised training with data from MultiWOZ and extra data from two other datasets
If you want to improve the performence of the model trained on MultiWOZ, you can run semi-supervised experiments over MultiWOZ and two other datasets.

After supervised training over all data from MultiWOZ, you can conduct self training:
```
bash train_ST_aug.sh
```
or semi variational learning:
```
bash train_VL_aug.sh
```

## Evaluation 
To test the performance of your model on the test set of MultiWOZ:
```
bash test.sh
```
   
## Checkpoint
We provide our SOTA models on https://pan.baidu.com/s/1MeFypGOwufAeLkUEQVqItQ extraction code: wp3y. 

The supervised baseline model is trained over MultiWOZ2.1, and the other two models (LS-GPT+Semi-VL and LS-GPT+Semi-ST) are trained over MultiWOZ2.1 and unlabeled dialogs from other datasets. 

Download these checkpoints to experiments_21, and change the path in `test.sh`, then you can get results as follows:
|Model|Inform|Success|BLEU|Combined|
|-----|------|------|-----|--------|
|baseline|90.19 |77.78 |15.59 |99.58|
|LS-GPT+Semi-ST|90.79 |80.38 |16.02 |101.61 |
|LS-GPT+Semi-VL|91.49 |79.38 |15.54 |100.98 |

