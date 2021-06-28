# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script provides two functions for visualization of 
# results and more accurate computation of joint goal
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_semi import Modal, parse_arg_cfg, fix_cfg
import argparse
from config import global_config as cfg
import random
import json
import time
import logging
import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from compute_joint_acc import compute_jacc
class Semi_Model(Modal):
    def __init__(self, device) -> None:
        super().__init__(device=device)
        self.device=device[0]
        self.output_num=3

    def generate_label_for_extra(self):
        st=time.time()
        data1=self.load_data('../data/taskmaster/TM2MultiWOZ.json')
        data2=self.load_data('../data/schema-guided/SGD2MultiWOZ.json')
        cfg.batch_size=cfg.eval_batch_size
        batches1=self.reader.get_batches('test',data=data1)
        batches2=self.reader.get_batches('test',data=data2)
        random.shuffle(batches1)
        random.shuffle(batches2)
        new_data1={}
        new_data2={}
        cfg.gen_db=True
        for batch in batches1:
            if len(batch[0])>16 or len(batch[0])<2:
                continue
            new_batch=self.gen_batch_bspn(batch)
            for encoded_dial in new_batch:
                new_dial=[]
                for encoded_turn in encoded_dial:
                    new_turn={}
                    for key in ['user','resp','bspn','db','aspn']:
                        new_turn[key]=self.tokenizer.decode(encoded_turn[key])
                    if self.output_num>0:
                        print(new_turn)
                        self.output_num-=1
                    new_dial.append(new_turn)
                new_data1[encoded_turn['dial_id']]=new_dial
        json.dump(new_data1, open('TM_temp1.json', 'w'), indent=2)
        
        for batch in batches2:
            if len(batch[0])>16 or len(batch[0])<2:
                continue
            new_batch=self.gen_batch_bspn(batch)
            for encoded_dial in new_batch:
                new_dial=[]
                for encoded_turn in encoded_dial:
                    new_turn={}
                    for key in ['user','resp','bspn','db','aspn']:
                        new_turn[key]=self.tokenizer.decode(encoded_turn[key])
                    new_dial.append(new_turn)
                new_data2[encoded_turn['dial_id']]=new_dial

        json.dump(new_data2, open('SGD_temp1.json', 'w'), indent=2)
        print('inference time:{}'.format(time.time()-st))

    def compute_standard_joint_goal(self):
        baseline_path=['/home/liuhong/UBAR/experiments_21/all_full_aug1.2_sd11_lr0.0001_bs2_ga16/best_score_model',
        '/home/liuhong/UBAR/experiments_21/all_ST_fix_50_act_sd11_lr5e-05_bs2_ga16/best_score_model'
        ]
        '''
        for proportion in ['10','20','30','40','50']:
            baseline_path.append('./experiments_21/all_pre_{}_act_sd11_lr0.0001_bs2_ga16/best_score_model'.\
                format(proportion))
            baseline_path.append('./experiments_21/all_ST_fix_{}_act_sd11_lr0.0001_bs2_ga16/best_score_model'.\
                format(proportion))
            baseline_path.append('./experiments_21/all_VL_{}_act_sd11_lr0.0001_bs2_ga16/best_score_model'.\
                format(proportion))
        '''
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test')
        if cfg.debugging:
            batches=batches[:2]
        for path in baseline_path:
            '''
            if not os.path.exists(path):
                continue
            '''
            self.model=GPT2LMHeadModel.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            max_len=60#for additional dataset, we don't generate too long
            sos_b_id=self.sos_b_id
            eos_b_id=self.eos_b_id
            eos_a_id=self.eos_a_id
            result_collection={}
            st=time.time()
            with torch.no_grad():
                for batch in batches:
                    try:
                        batch_size=len(batch)
                        contexts=[[] for i in range(len(batch))]
                        resp=[[] for i in range(len(batch))]
                        aspn_gen=[[] for i in range(len(batch))]
                        bs_gen=[]
                        for turn_num in range(len(batch[0])):
                            past_key_values=None
                            end_flag=np.zeros(len(batch))
                            contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,\
                                prior=True,resp_gen=resp,aspn_gen=aspn_gen)
                            inputs,attentions=self.batch_align(contexts,left_len=max_len,return_attn=True)
                            inputs=torch.tensor(inputs).to(self.device)
                            attentions=torch.tensor(attentions).to(self.device)
                            if self.global_output>0 and cfg.example_log:
                                print('generation examples:')
                                print(self.tokenizer.decode(contexts[0]))
                                self.global_output-=1
                            for i in range(max_len):
                                position_ids = attentions.long().cumsum(-1) - 1
                                position_ids.masked_fill_(attentions == 0, 1)
                                if past_key_values is not None:
                                    position_ids=position_ids[:, -1].unsqueeze(-1)
                                outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,
                                    return_dict=True,use_cache=True,past_key_values=past_key_values)#B,T,V
                                past_key_values=outputs.past_key_values
                                if cfg.sample_type=='top1':
                                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                                elif cfg.sample_type=='topk':
                                    prob=F.softmax(outputs.logits[:,-1,:],dim=-1)#B,V
                                    topk_probs, topk_words = torch.topk(prob, cfg.topk_num)#B,topk_num
                                    widx = torch.multinomial(topk_probs, 1, replacement=True)#B,1
                                    preds = torch.gather(topk_words, 1, widx).squeeze()#B
                                if i==0:
                                    bs_tensor=preds.unsqueeze(1)
                                else:
                                    bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                                inputs=preds.unsqueeze(1)
                                attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(self.device)),dim=1)
                                end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                                if sum(end_flag==0)==0:
                                    break
                            bs_gen,db_gen=self.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)
                            contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,prior=True,db_gen=db_gen)

                            for i in range(len(batch)):
                                batch[i][turn_num]['bspn_gen']=bs_gen[i]
                                batch[i][turn_num]['db_gen']=db_gen[i]
                                resp[i]=batch[i][turn_num]['aspn']+batch[i][turn_num]['resp']#take aspn and resp as one resp
                        for dialog in batch:
                            result_collection.update(self.reader.inverse_transpose_turn(dialog))
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            logging.info("WARNING: ran out of memory during generation, and the batch will be divided half, batch size:{}, turn num:{}"\
                                .format(len(batch),len(batch[0])))
                            if hasattr(torch.cuda, 'empty_cache'):
                                with torch.cuda.device(self.device):
                                    torch.cuda.empty_cache()
                            #current batch out of memory, split it half
                            batches+= [ batch[:len(batch)//2], batch[len(batch)//2:] ]
                        else:
                            logging.info(str(exception))
                            raise exception
            results, field= self.reader.wrap_result_lm(result_collection)
            #field = ['dial_id', 'turn_num', 'user', 'resp', 'bspn', 'bspn_gen', 'aspn', 'aspn_gen','db_gen']
            joint_acc=compute_jacc(results)
            cfg.eval_load_path=path
            for result in results:
                if 'gtbs' in result or 'predbs' in result:
                    print(result)
            self.reader.save_result('w', results, field, write_title='DECODED RESULTS:',result_name='result2.csv')
            print(path)
            print('inference time:{}, joint_acc:{:.3f}'.format(time.time()-st, joint_acc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    parse_arg_cfg(args)
    if cfg.dataset==1:
        fix_cfg()
    cfg.mode=args.mode
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    # initialize model
    m = Semi_Model(cfg.cuda_device)
    #m.generate_label_for_extra()
    cfg._init_logging_handler(args.mode)
    m.compute_standard_joint_goal()

if __name__ == "__main__":
    main()