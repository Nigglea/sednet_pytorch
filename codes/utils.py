import pandas as pd
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

def label_append(data, event):
    columns = list(data.keys())
    for i,c in enumerate(columns):
        data[c].append(event[i])
    return data

def return_minmax(A:torch.Tensor):
    minA = A.min(dim=1)
    maxA = A.max(dim=1)

    return minA.values,maxA.values

def new_minmax(minA:torch.Tensor,maxA:torch.Tensor,minB:torch.Tensor,maxB:torch.Tensor):

    C = torch.cat([minA,minB])
    D = torch.cat([maxA,maxB])
    minC = C.min(dim=0)
    maxC = D.max(dim=0)
    return minC.values.view(1,-1),maxC.values.view(1,-1)

def apply_minmax(A:torch.Tensor,minA:torch.Tensor,maxA:torch.Tensor,device='cuda'):
    minA = minA.view(1,1,-1)
    maxA = maxA.view(1,1,-1)
    #print(minA.shape,maxA.shape)
    maxA = torch.cat([maxA for i in range(12920)],dim=-2)
    minA = torch.cat([minA for i in range(12920)],dim=-2)
    eps = 1e-8*torch.ones([1,12920,6]).to(device=device)
    print(A.shape,minA.shape,maxA.shape,eps.shape)
    numB = A-minA
    denB = maxA-minA+eps
    print(numB.shape,denB.shape)
    B=numB/denB
    return B

def return_mean_std(A:torch.Tensor):
    meanA = A.mean(dim=1)
    stdA = A.std(dim=1)

    return meanA,stdA

def new_mean_std(meanA:torch.Tensor,stdA:torch.Tensor,meanB:torch.Tensor,stdB:torch.Tensor):

    C = torch.cat([meanA,meanB])
    D = torch.cat([stdA,stdB])
    meanC = C.mean(dim=0)
    stdC = D.mean(dim=0)
    return meanC.view(1,-1),stdC.view(1,-1)

def apply_mean_std(A:torch.Tensor,meanA:torch.Tensor,stdA:torch.Tensor,device='cuda'):
    meanA = meanA.view(1,1,-1)
    stdA = stdA.view(1,1,-1)
    #print(meanA.shape,stdA.shape)
    meanA = torch.cat([meanA for i in range(12920)],dim=-2)
    stdA = torch.cat([stdA for i in range(12920)],dim=-2)
    eps = 1e-12*torch.ones([1,12920,6]).to(device=device)
    print(A.shape,meanA.shape,stdA.shape,eps.shape)
    numB = A-meanA
    denB = stdA+eps
    print(numB.shape,denB.shape)
    B=numB/denB
    return B
