# Librerias
#%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
print(torch.cuda.is_available())
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import datetime
from codes import utils
from codes import metric as mt
from codes.dataset2 import WetSoundDataset
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.nn.functional import sigmoid
from codes.models import *
from torch.utils.tensorboard import SummaryWriter
#%load_ext tensorboard

# %%
os.system('rm -rf ./tmp/')
main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

FOLDER_ANNOTATIONS = "datos/"
N_FOLDS = 4
AUDIO_DIR = os.path.join(main,"features/train_set/stai/")
SAMPLE_RATE = 44100
LEN_SEC = 300
LEN_SAMPLES = LEN_SEC*SAMPLE_RATE
NUM_SAMPLES = 256   # 1292 #Cantidad de secuencias , no largo de secuencia 12920/40
categories = pd.read_csv(FOLDER_ANNOTATIONS+'categories.txt',names=['class'])
__class_labels = {str(c[0]):i for i,c in enumerate(categories.values)}
N_FFT = 2048
HOP = int(N_FFT/2)
LEN_MBE = round(LEN_SAMPLES/int(HOP*NUM_SAMPLES))
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device {device}")






wet = WetSoundDataset(AUDIO_DIR,
                        LEN_MBE,
                        NUM_SAMPLES,
                        __class_labels,
                        device)
print(f"There are {len(wet)} samples in the dataset.")




model = STAInet(NUM_SAMPLES,n_class=len(__class_labels))
print(model)

max_epochs = 1000
kfold = KFold(n_splits=N_FOLDS, shuffle=True)

# K-fold Cross Validation model evaluation
#%tensorboard --logdir runs
for fold, (train_ids, val_ids) in enumerate(kfold.split(wet)):
    torch.manual_seed(12345) # Inicialización
    # Print
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    kfold_trainset = Subset(wet,train_ids)
    kfold_valset = Subset(wet, val_ids)

    trainloader = DataLoader(
                    kfold_trainset, 
                    batch_size=64, shuffle=True)
    valloader = DataLoader(
                    kfold_valset,
                    batch_size=128,shuffle=False)
    ntrain, nval = len(trainloader),len(valloader)
    
    
    
    model = STAInet(NUM_SAMPLES,n_class=len(__class_labels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3,weight_decay=1e-6,eps=1e-8)
    criterion = torch.nn.BCELoss(reduction="mean")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    writer = SummaryWriter('runs/stainet_trainer_{}'.format(timestamp))
    epoch_number = 0
    clip = 0.5
    best_f1 = 0.
    best_er = 1e15
    best_vloss = 1e15
    F1 = False
    loss = False if F1 else True
    ER = False if (F1 or loss) else True
    cnt = 0
    patience = int(max_epochs*0.1)
    if F1:
        patience_parameter = 'F1 Score'
    if loss:
        patience_parameter = 'Validation Loss'
    if ER:
        patience_parameter = 'Error Rate'

    print('This model will be train with {} patience'.format(patience_parameter))

    for epoch in range(max_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        running_tloss = 0.0
        for i,tdata in enumerate(trainloader):
            tinputs,tlabels = tdata 
            optimizer.zero_grad()
            toutputs = model.forward(tinputs)
            tloss = criterion(toutputs,tlabels)
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip,norm_type=2.)
            optimizer.step()
            running_tloss +=tloss.item()
        avg_tloss = running_tloss /(i+1)

        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.0
        vf1 = 0.
        ver = 0.
        with torch.no_grad():
            for j, vdata in enumerate(valloader):
                vinputs, vlabels = vdata
                voutputs = model.forward(vinputs)
                vloss = criterion(voutputs, vlabels)
                j_vf1 = mt.f1_score(voutputs,vlabels)
                j_ver = mt.error_rate(voutputs,vlabels)
                running_vloss += vloss
                vf1 += j_vf1
                ver += j_ver
        avg_vloss = running_vloss / nval
        vf1 = vf1/nval
        ver = ver/nval
        print('LOSS train {} valid {}'.format(avg_tloss, avg_vloss))
        torch.cuda.empty_cache()
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_tloss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('F1 - score',{'F1': vf1},epoch_number+1)
        writer.add_scalars('Error Rate',{'error_rate': ver},epoch_number+1)
        writer.flush()

        # Track best performance, and save the model's state
        if F1:
            if vf1 > best_f1:
                best_f1 = vf1
                print('Best Epoch is {}'.format(epoch_number+1))
                model_path = 'models/stai_model_{}.pt'.format(timestamp)
                if device == 'cuda':
                    model.cpu()
                torch.save(model.state_dict(), model_path)
                cnt=0
                model.to(device)
            else:
                cnt += 1
        if ER:
            if ver < best_er:
                best_er = ver
                print('Best Epoch is {}'.format(epoch_number+1))
                model_path = 'models/stai_model_{}.pt'.format(timestamp)
                if device == 'cuda':
                    model.cpu()
                torch.save(model.state_dict(), model_path)
                cnt = 0
                model.to(device) 
            else:
                cnt += 1
                print('Best Epoch was ',epoch_number-cnt+1)
                print('{} Epochs left for Early Stopping'.format((patience-cnt)))
        if loss:
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                print('Best Epoch is {}'.format(epoch_number+1))
                model_path = 'models/stai_model_{}.pt'.format(timestamp)
                if device == 'cuda':
                    model.cpu()
                torch.save(model.state_dict(), model_path)
                cnt = 0
                model.to(device) 
            else:
                cnt += 1
                print('Best Epoch was ',epoch_number-cnt+1)
                print('{} Epochs left for Early Stopping'.format((patience-cnt)))
            
        
        if cnt == patience:
            print('Early Stopping...')
            break

        epoch_number += 1
    
torch.cuda.empty_cache()



