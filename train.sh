# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
posterior=False
python train_semi.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=50\
    cuda_device=0\
    model_act=True\
    dataset=1\
    delex_as_damd=True\
    gen_db=False\
    posterior_train=$posterior
