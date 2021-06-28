# LS-GPT
## Requirements
* python=3.6
* pytorch=1.2.0
* trasformers=3.5.1
## Training
Supervised training over all data (labeled) in MultiWOZ:
```
bash train.sh
```
Semi-supervised training with partial labeled data in MultiWOZ
If you haven't pretrain the model on labeled data:
```
bash pretrain.sh
```
  Self Training:
```
bash train_ST.sh
```
  Semi Variational Learning:
```
bash train_VL.sh
```
If you want to improve the performence of the model trained on MultiWOZ, you can run semi-supervised experiments over MultiWOZ and two other datasets.
   Self Training:
```
bash train_ST_aug.sh
```
   Semi Variational Learning:
```
bash train_VL_aug.sh
```
## Evaluation 
To test the performance of your model on the test set of MultiWOZ:
```
bash test.sh
```

* Note: You can change the parameters such as cuda_device in all the sh file to satisfy your requirement.
   
