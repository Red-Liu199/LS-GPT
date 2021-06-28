
# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script runs the Semi-VL (variational learning) experiment on MultiWOZ and two other datasets.
# Data from MultiWOZ is labeled and data from other datasets is unlabeled. 
# Before Semi-VL, make sure that you have pretrained the models on all the dialogs in MultiWOZ.

path1=/home/liuhong/UBAR/experiments_21/all_full_510_sd11_lr0.0001_bs2_ga16/best_score_model
path2=/home/liuhong/UBAR/experiments_21/all_full_pos510_sd11_lr0.0001_bs2_ga16/best_score_model
python train_semi.py\
    -mode semi_train\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=50\
    cuda_device=0,1\
    data_aug=True\
    model_act=True\
    dataset=1\
    delex_as_damd=True\
    use_scheduler=True\
    exp_no=VL-standard\
    only_SGD=False
