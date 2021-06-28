import json
from transformers import GPT2Tokenizer
import re
import copy
informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
replace_slots={
    "taxi": ["leave", "arrive"],
    "police": [],
    "hospital": [],
    "hotel": ["name"],
    "attraction": ["name"],
    "train": ["arrive", "leave"],
    "restaurant": ["name", "time"]
}
path='/home/liuhong/UBAR/data.json'
save_path='test_data.json'
data=json.load(open(path,'r', encoding='utf-8'))
test_data=[]
#time_pattern=re.compile(r'[0-9][0-9]{1,2}:[0-9][0-9]{2}')
name_pool=[]
time_pool=[]
moment_pool=[]
map1=json.load(open('data/multi-woz-2.1-processed/delex_multi_valdict.json','r', encoding='utf-8'))
map2=json.load(open('data/multi-woz-2.1-processed/delex_single_valdict.json','r', encoding='utf-8'))
for key in map1:
    if map1[key]=='name' and key not in name_pool:
        name_pool.append(key)
    if map1[key] in ['arrive','leave'] and key not in moment_pool:
        moment_pool.append(key)
    if map1[key]=='time' and key not in time_pool:
        time_pool.append(key)
for key in map2:
    if map2[key]=='name' and key not in name_pool:
        name_pool.append(key)
    if map2[key] in ['arrive','leave'] and key not in moment_pool:
        moment_pool.append(key)
    if map2[key]=='time' and key not in time_pool:
        time_pool.append(key)
print(len(name_pool),len(time_pool),len(moment_pool))
for dial in data['test']:
    new_dial=copy.deepcopy(dial)
    flag=0
    for turn_id, turn in enumerate(dial):
        user=copy.deepcopy(turn['user'])
        for name in name_pool:
            if name in user:
                user=re.sub(name, 'tsinghua', user)
                flag=1
        for time in time_pool:
            if time in user:
                user=re.sub(time, '99 minutes', user)
                flag=1
        for moment in moment_pool:
            if moment in user:
                user=re.sub(moment, '11:11', user)
                flag=1
        new_dial[turn_id]['user']=user
        new_dial[turn_id]['dial_id']=turn['dial_id']+'_new'
    if flag:
        test_data.append(dial)
        test_data.append(new_dial)
print('total dials:',len(test_data))
json.dump({'test':test_data}, open(save_path, 'w'), indent=2)
'''
tokenizer=GPT2Tokenizer.from_pretrained('/home/liuhong/UBAR/experiments_21/supervised_baseline/best_score_model')
path='/home/liuhong/UBAR/data/multi-woz-processed/new_db_se_blank_encoded.data.json'
save_path='data.json'
encoded_data=json.load(open(path,'r', encoding='utf-8'))
data={}
for set in encoded_data:
    data[set]=[]
    for dial in encoded_data[set]:#dial是列表，turn是字典
        new_dial=[]
        for turn in dial:
            new_turn={}
            for key in turn:
                if key in ['user','usdx','resp','bspn','bsdx','aspn','dspn','db']:
                    new_turn[key]=tokenizer.decode(turn[key])
                else:
                    new_turn[key]=turn[key]
            new_dial.append(new_turn)
        data[set].append(new_dial)
json.dump(data, open(save_path, 'w'), indent=2)
'''