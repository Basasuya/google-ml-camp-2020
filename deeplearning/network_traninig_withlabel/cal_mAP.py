import os
import h5py
import numpy as np
import argparse
import json
import re
from sklearn.neighbors import NearestNeighbors

def get_result(q, train_embedding, train_file_list):
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(train_embedding)
    return neigh.kneighbors(q, return_distance=False)[0]

def cal_AP(query_file, res_list):
    num_of_correct = 0
    sum_of_precision = 0.0
    for i, name in enumerate(res_list):
        if name[:get_first_digit_pos(name) - 1] == query_file[:get_first_digit_pos(query_file) - 1] :
            num_of_correct += 1
            sum_of_precision +=  num_of_correct / (i + 1.)
        return sum_of_precision / num_of_correct if num_of_correct > 0 else 0

def cal_mAP(query_file_list, train_file_list, train_embedding, query_embedding):
    sum_of_AP = 0.
    for idx, f in enumerate(query_file_list):
        if idx % 30 == 0:
            print(idx, "out of 185 calculated.")
        q = query_embedding[query_file_list.index(f)]
        res_list = [train_file_list[i] for i in get_result([q], train_embedding, train_file_list)]
        sum_of_AP += cal_AP(f, res_list)
    return sum_of_AP / len(query_file_list)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-train_embedding", required = True,
        help = "Name of train embedding")
    ap.add_argument("-query_embedding", required = True,
        help = "Name of query embedding")
    ap.add_argument("-train_json", required = True,
        help = "Name of train json")
    ap.add_argument("-query_json", required = True,
        help = "Name of query json")
    args = vars(ap.parse_args())
    
    train_embedding_file = args["train_embedding"]
    train_json_file = args["train_json"]

    query_embedding_file = args["query_embedding"]
    query_json_file = args["query_json"]

    train_embedding = np.load(train_embedding_file)['ans']
    query_embedding = np.load(query_embedding_file)['ans']

    with open(train_json_file, 'r') as f:
        name = json.load(f)
        train_file_list = name['name']

    with open(query_json_file, 'r') as f:
        name = json.load(f)
        query_file_list = name['name']

    print("mAP is ", cal_mAP(query_file_list, train_file_list, train_embedding, query_embedding))

