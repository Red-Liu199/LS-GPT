# LS-GPT

Supervised training over all data (labeled) in MultiWOZ:
   bash train.sh

Semi-supervised training with partial labeled data in MultiWOZ
  Self Training:
     If you haven't pretrain the model on labeled data:
        bash pretrain.sh
     bash train_ST.sh

  Semi Variational Learning:
     If you haven't pretrain the model on labeled data:
        bash pretrain.sh
     bash train_VL.sh
   
If you want to improve the performence of the model trained on MultiWOZ, you can run semi-supervised experiments over MultiWOZ and two other datasets.
   Self Training:
      bash train_ST_aug.sh
   Semi Variational Learning:
      bash train_VL_aug.sh
      
To test the performance of your model on the test set of MultiWOZ:
   bash test.sh
Note: You can change the parameters such as cuda_device in the sh file to satisfy your requirement.
   
