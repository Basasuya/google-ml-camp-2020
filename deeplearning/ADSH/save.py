import json
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Save data")
parser.add_argument('--filename', required=True, help="Path to pkl is indexed")
opt = parser.parse_args()
filename = opt.filename

with open(filename,'rb') as f:
    data = pickle.load(f)
hashcode = np.concatenate((data['rB'], data['qB']), axis=0)
np.savez('hashcode.npz',ans=hashcode)

file_names, labels = [], []
with open('./file_names.txt', 'r') as f:
    for line in f:
        file_names.append(line.strip())
with open('./labels.txt', 'r') as f:
    for line in f:
        labels.append(int(line.strip()))
train_index, test_index = train_test_split(range(len(file_names)), test_size=0.2, random_state=42, stratify=labels)
indice = train_index + test_index
image_names = [file_names[i] for i in indice]
hash_json = {'name': image_names, 'label': indice}
with open('./hash_json.json', 'w', encoding='utf-8') as json_file:
    json.dump(hash_json, json_file)
    print("write json file success!")
