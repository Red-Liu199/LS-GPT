# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script tests the performance(BLEU, inform, success and combined score) of the model you choose

path=experiments_21/all_full_aug_VL_sd11_lr0.0001_bs2_ga16/best_score_model
python train_semi.py -mode test\
    -cfg gpt_path=$path  cuda_device=0  \
    fast_validate=True\
    model_act=True\
    dataset=1
