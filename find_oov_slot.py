from transformers import GPT2Tokenizer
import json
#from  ontology import *

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
path='/home/liuhong/UBAR/data20.json'
data=json.load(open(path,'r', encoding='utf-8'))
slots={}
for dial in data['pre_data']:
    for turn in dial:
        bsdx=turn['bsdx']
        bsdx=bsdx.split()[1:-1]
        for word in bsdx:
            if word.startswith('['):
                domain=word[1:-1]
                if domain not in slots:
                    slots[domain]=[]
            else:
                if word not in slots[domain]:
                    slots[domain].append(word)
print('dials:',len(data['pre_data']))
print(slots)
for key in slots:
    if len(slots[key])!=len(informable_slots[key]):
        print(key,'\n',slots[key])
'''
tokenizer=GPT2Tokenizer.from_pretrained('/home/liuhong/UBAR/experiments_21/supervised_baseline/best_score_model')
path='/home/liuhong/UBAR/data/multi-woz-2.1-processed/divided_data20.json'
save_path='data20.json'
encoded_data=json.load(open(path,'r', encoding='utf-8'))
data={}
for set in ['pre_data']:
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

