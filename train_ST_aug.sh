# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script runs the Semi-ST (self training) experiment on MultiWOZ and two other datasets.
# Data from MultiWOZ is labeled and data from other datasets is unlabeled. 
# Before Semi-ST, make sure that you have pretrained the models on all the dialogs in MultiWOZ.

path = experiments_21/supervised_baseline/best_score_model
exp_no = ST_aug 
python train_semi.py\
    -mode semi_ST\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=30\
    cuda_device=0\
    gpt_path=$path\
    data_aug=True\
    model_act=True\
    dataset=1\
    len_limit=True\
    use_scheduler=True\
    exp_no=$exp_no\
    only_SGD=True