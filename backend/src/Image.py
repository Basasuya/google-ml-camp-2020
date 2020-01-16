import numpy as np
import pandas as pd
import json
from tulip import tlp
import random
import numpy as np
from sklearn.manifold import TSNE
import umap

from queue import PriorityQueue as PQ

class Image:
    def __init__(self, embedding='../data/vgg_raw/embedding.npz', detail="../data/vgg_raw/name.json", data="../data/vgg_raw/data.json"):
        self.embedding=embedding
        self.detail=detail
        self.data=data
        self.embed = np.load(self.embedding)["ans"]

        print("data shape: ", self.embed.shape)
        self.points = len(self.embed)
    
    def getDimension(self):
        with open(self.data, 'r') as f:
            detail = json.load(f)
        return detail

    def constructKnn(self):
        pass
    
    def getKnn(self, id, k):
        searchId = self.embed[id]
        pq = PQ()
        for i in range(self.points):
            if(i == id):
                continue
            else:
                pq.put((-np.sum( (searchId - self.embed[i]) * (searchId - self.embed[i]) ), i))
                if(pq.qsize() > k):
                    pq.get()
        result = []
        while(pq.empty() == False):
            tmp = pq.get()
            result.append([tmp[0], tmp[1]])
        result = result[: :-1] 
        for i in range(k):
            result[i][0] = -result[i][0]
        # print(result)
        ans = []
        for i in range(k):
            ans.append(result[i][1])
        return { "result": ans }

    def prepare(self):
        # embed = np.load(self.embedding)["ans"]
        low_emb = embedding = umap.UMAP(n_neighbors=100, metric='euclidean', verbose = True).fit_transform(self.embed)
        
        with open(self.detail, 'r') as f:
            detail = json.load(f)
        # print(detail)
        result = []
        for i in range(len(detail["name"])):
            result.append({"positionX" : float(low_emb[i][0]), "positionY" : float(low_emb[i][1]), "path": detail["name"][i], "key" : i})
            if(detail.__contains__("label")):
                result[-1]["label"] = detail["label"][i]

        with open(self.data, 'w') as f:
            json.dump(result, f)




if __name__ == "__main__":
    choose = "vgg_label"
    A = Image(embedding="../data/" + choose + "/embedding.npz", detail="../data/" + choose + "/name.json", data="../data/" + choose + "/data.json")
    A.prepare()   
    # A.getKnn(2, 3) 