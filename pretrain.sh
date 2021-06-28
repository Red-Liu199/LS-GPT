# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script pretrains model on the labeled data of MultiWOZ.
# You can choose any supervised proportion (ratio).
# If posterior is True, you'll pretrain the inference model
# else you'll pretrain the generative model.
ratio=20
posterior=False
python train_semi.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=50\
    cuda_device=0\
    spv_proportion=$ratio\
    model_act=True\
    dataset=1\
    posterior_train=$posterior