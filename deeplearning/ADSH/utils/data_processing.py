import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle


# *********************** need modification ***********************
# *****************************************************************
# class DatasetProcessing(Dataset):
#     def __init__(self, dirpath, filename):
#         """Init the class."""
#         pass
#
#     def __getitem__(self, index):
#         """Get an item with a giving index.
#         Args:
#             index: int
#         Returns:
#             a list in the form of [img, label, index].
#             img: PIL.image
#             label: int
#             index: int
#         """
#         pass
#
#     def __len__(self):
#         """Return the number of samples."""
#         pass
# *******************************************************************
# *******************************************************************

class DatasetProcessing(Dataset):
    def __init__(self, data_path, datafile, labelfile, indice, transform=None):
        self.data_path = data_path
        filenames, labels = [], []
        datapath = os.path.join('./', datafile)
        labelpath = os.path.join('./', labelfile)
        self.transform = transform
        with open(datapath, 'r') as f:
            for line in f:
                filenames.append(line.strip())
        with open(labelpath, 'r') as f:
            for line in f:
                labels.append(int(line.strip()))
        print("len of filenames", len(filenames))
        print("len of labels", len(labels))
        self.filenames = [filenames[i] for i in indice]
        self.labels = [labels[i] for i in indice]

    def __getitem__(self, index):
        filename = self.filenames[index]
        filepath = os.path.join(self.data_path, filename)
        img = Image.open(filepath)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor(self.labels[index])
        return img, label, index

    def __len__(self):
        return len(self.filenames)
