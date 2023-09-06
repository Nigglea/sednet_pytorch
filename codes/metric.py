import torch
import numpy as np
from codes import utils
import sklearn.metrics
import pandas as pd

def save_batch_prediction(filenames,pred,categories,df=None):
    
    class_label = categories
    columns = ["event_label","onset","offset","filename"]
    data = {c:[] for c in columns}
    if df is None:
        pred_df = pd.DataFrame(columns=columns)
    else:
        pred_df = df
    T = np.linspace(0,300,pred.shape[2])
    for f,filename in enumerate(filenames):
        pred_class = pred[f,:,:]
        for i,p in enumerate(pred_class):
            label = class_label[i]
            c=0
            for j,t in enumerate(p):
                if t:
                    if c==0:
                        t_ini = np.around(T[j],3)
                        c += 1
                    if j == len(p)-1:
                        t_end = np.around(T[j],3)
                        event = [label,t_ini,t_end,filename]
                        data = utils.label_append(data,event)            
                else:
                    if c>0:
                        t_end = np.around(T[j],3)
                        event = [label,t_ini,t_end,filename]
                        data = utils.label_append(data,event)
                        c = 0
    event_labels = pd.DataFrame(data)
    pred_df = pd.concat([pred_df,event_labels])
    return pred_df

def save_prediction(filename,pred,categories,df=None):
    
    class_label = categories
    columns=["event_label","onset","offset","filename"]
    if df is None:
        pred_df = pd.DataFrame(columns=columns)
    else:
        pred_df = df
    pred_class = pred
    T = np.linspace(0,300,pred_class.shape[1])
    for i,p in enumerate(pred_class):
        c=0
        for j,t in enumerate(p):
            if t:
                if c==0:
                    t_ini = np.around(T[j],3)
                    c+=1
                if j == len(p)-1:
                    t_end = np.around(T[j],3)
                    event = pd.DataFrame([class_label[i],t_ini,t_end,filename],index = columns)
                    pred_df = pd.concat([pred_df,event.T],axis = 0,ignore_index=True)
            else:
                if c>0:
                    t_end = np.around(T[j],3)
                    event = pd.DataFrame([class_label[i],t_ini,t_end,filename],index = columns)
                    pred_df = pd.concat([pred_df,event.T],axis = 0,ignore_index=True)
                    c=0
            
    
    return pred_df


eps = torch.finfo(torch.float).eps

def accuracy(y, label):
    if label.ndim == 3: # SED
        return torch.sum((y > 0.5) == label).item()/(label.shape[1]*label.shape[2])
    else:
        return torch.sum(y.argmax(1) == label).item()

def f1_score(y, label):
    if label.ndim == 3: # SED
        frames_in_1_sec = 100
        y = torch.where(y > 0.5, 1., 0.)
        f1_score = np.round(f1_overall_1sec(y, label, frames_in_1_sec), 3)
        return f1_score
    else:
        return sklearn.metrics.f1_score(label.cpu().numpy(), y.cpu().argmax(dim=1).numpy(), average='macro')

def error_rate(y, label):
    if label.ndim == 3: # SED
        frames_in_1_sec = 100
        y = torch.where(y > 0.5, 1., 0.)
        error_rate = np.round(er_overall_1sec(y, label, frames_in_1_sec).item(), 3)
        return error_rate
    else:
        return eps

def reshape_3Dto2D(A):
    return A.view(A.shape[0] * A.shape[1], A.shape[2])

def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score

def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = torch.logical_and(T == 0, O == 1).sum(1)
    FN = torch.logical_and(T == 1, O == 0).sum(1)
    S = torch.min(FP, FN).sum()
    D = torch.max(torch.tensor([0]), FN-FP).sum()
    I = torch.max(torch.tensor([0]), FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + torch.tensor([0])+eps)
    return ER

def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = torch.zeros((new_size, O.shape[1]))
    T_block = torch.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = torch.max(O_block_i, axis=0)[0]
        T_block[i, :] = torch.max(T_block_i, axis=0)[0]
    return f1_overall_framewise(O_block, T_block)

def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = torch.zeros((new_size, O.shape[1]))
    T_block = torch.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = torch.max(O_block_i, axis=0)[0]
        T_block[i, :] = torch.max(T_block_i, axis=0)[0]
    return er_overall_framewise(O_block, T_block)