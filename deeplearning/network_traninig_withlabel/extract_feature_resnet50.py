import os
import h5py
import numpy as np
import argparse
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.applications.resnet_v2 import preprocess_input

label_list = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue','Siamese','Sphynx','american_bulldog','american_pit_bull_terrier','basset_hound','beagle','boxer','chihuahua','english_cocker_spaniel','english_setter','german_shorthaired','great_pyrenees','havanese','japanese_chin','keeshond','leonberger','miniature_pinscher','newfoundland','pomeranian','pug','saint_bernard','samoyed','scottish_terrier','shiba_inu','staffordshire_bull_terrier','wheaten_terrier','yorkshire_terrier']

ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
    help = "Path to database which contains images to be indexed")
ap.add_argument("-embedding", required = True,
    help = "Name of output embedding")
ap.add_argument("-json", required = True,
    help = "Name of json")
args = vars(ap.parse_args())

import re

def get_first_digit_pos(s):
    first_digit = re.search('\d', s)
    return first_digit.start()

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

from numpy import linalg
def extract_from_img(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = extract_model.predict(img)
        norm_feat = feat[0] / linalg.norm(feat[0])
        return norm_feat

if __name__ == "__main__":

    db = args["database"]
    embedding_file = args["embedding"]
    json_file = args["json"]
    img_list = get_imlist(db)
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []
    labels = []

    model = load_model('/home/cyhong021/saved_model/resnet50/model_epoch100_loss0.05_acc1.00.h5')
    
    layer_name = 'dense'
    extract_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

    for i, img_path in enumerate(img_list):
        norm_feat = extract_from_img(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        #print(names)
        #print(label_list.index(img_name[:get_first_digit_pos(img_name) - 1]))
        labels.append(label_list.index(img_name[:get_first_digit_pos(img_name) - 1]))
        if i%300 == 0:
            print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))


    with open(json_file, 'w') as outfile:
        json.dump({'name': names, 'label':labels}, outfile)
    feats = np.array(feats)
    print(feats.shape)
    np.savez(embedding_file, ans=feats)
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

