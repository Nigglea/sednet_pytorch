import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def load_desc_file(_desc_file,__class_labels):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split(' ')
        name = words[0]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[1]), float(words[2]), __class_labels[words[-1]]])
    return _desc_dict

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

def create_set(elements, reference):
    df = pd.DataFrame(columns=reference.columns)
    for e in elements:
        data = reference[reference.filenames==e]
        df = df.append(data,ignore_index=True)
    return df

def num_2wet(df,num):
    #humedal
    for i in range(len(df)):
        if df.iloc[i,2]<= num:
            if df.iloc[i,3]>= num:
                return df.iloc[i,0]