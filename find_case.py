# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
import json
import csv
sup_only_path='/home/liuhong/UBAR/experiments_21/all_pre_20_act_sd11_lr0.0001_bs2_ga16/best_score_model/result1.csv'
semi_ST_path='/home/liuhong/UBAR/experiments_21/all_ST_fix_20_act_sd11_lr0.0001_bs2_ga16/best_score_model/result1.csv'
semi_VL_path='/home/liuhong/UBAR/experiments_21/all_VL_20_act_sd11_lr0.0001_bs2_ga16/best_score_model/result1.csv'
f1=open(sup_only_path, 'r')
f2=open(semi_ST_path, 'r')
f3=open(semi_VL_path, 'r')
reader1=csv.reader(f1)
reader2=csv.reader(f2)
reader3=csv.reader(f3)
for n,(line1,line2,line3) in enumerate(zip(reader1,reader2,reader3)):
    if n>0:
        if n==1:
            key_map={}
            for i,key in enumerate(line1):
                key_map[key]=i
        elif line1[key_map['user']]!='':
            gt_bs=line1[key_map['bspn']]
            bs_gen1=line1[key_map['bspn_gen']]
            bs_gen2=line2[key_map['bspn_gen']]
            bs_gen3=line3[key_map['bspn_gen']]
            if bs_gen3==gt_bs and bs_gen1!=gt_bs and bs_gen2!=gt_bs and bs_gen1!=bs_gen2:
                print(line1[key_map['dial_id']])
f1.close()
f2.close()
f3.close()




