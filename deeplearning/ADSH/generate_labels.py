import os


data_path = r'./images'
file_list = [f for f in os.listdir(data_path) if f[-3:] == 'jpg']
label_set = set(['_'.join(name.split('_')[:-1]) for name in file_list])
d = {c:i for i, c in enumerate(label_set)}
with open('./labels.txt', 'a') as f:
    for file in file_list:
        f.write(str(d['_'.join(file.split('_')[:-1])])+'\n')

with open('./file_names.txt', 'a') as f:
    for file in file_list:
        f.write(file+'\n')

