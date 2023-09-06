# %%
# Librerias
#%matplotlib notebook
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import torchaudio
from codes import utils, metric
from codes.models import SEDnet
from codes.dataset2 import WetSoundDataset
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import shutil
# %% [markdown]
# # Test

# %%
main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# %%
FOLDER_ANNOTATIONS = "datos/"
AUDIO_DIR = os.path.join(main,"features/test_set/")
SAMPLE_RATE = 44100
LEN_SEC = 300
LEN_SAMPLES = LEN_SEC*SAMPLE_RATE
NUM_SAMPLES = 12920  #Cantidad de secuencias , no largo de secuencia 12920/40
categories = pd.read_csv(FOLDER_ANNOTATIONS+'categories.txt',names=['class'])
__class_labels = {str(c[0]):i for i,c in enumerate(categories.values)}
categ = {i:str(c[0]) for i,c in enumerate(categories.values)}
N_FFT = 2048
HOP = int(N_FFT/2)
N_MELS = 32
LEN_MBE = round(LEN_SAMPLES/int(HOP*NUM_SAMPLES))
frames_1_sec = int(SAMPLE_RATE/(N_FFT/2.0))
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device {device}")

wet = WetSoundDataset(AUDIO_DIR,
                        LEN_MBE,
                        NUM_SAMPLES,
                        __class_labels,
                        device,
                        test=True)
print(f"There are {len(wet)} samples in the dataset for test.")


lin = True
n_thresholds = 20
if lin:
    thresholds = np.round(np.linspace(0,1,n_thresholds+1),2)
if not lin:
    thresholds = np.round(1. - np.exp(np.linspace(np.log(1 - 0.99999), np.log(1 - 0.1), n_thresholds+1))[::-1],3)

path_models = "models/"
results = dict()
for fold_model in os.listdir(path_models):
    filter_file = fold_model.split(".")
    if len(filter_file)>1:
        if filter_file[1]=="pt":
            model = SEDnet(NUM_SAMPLES,len(__class_labels))
            model.load_state_dict(torch.load(path_models+fold_model))
            model.to(device)

            os.makedirs(filter_file[0]+'_pred')

            pred = list()
            targets = list()
            i=0
            
            testloader = DataLoader(wet,batch_size=8,shuffle=False)

            for th in thresholds:
                print('Threshold', th, end='\n')
                for j,aud in enumerate(testloader):
                    print(round(((j+1)/len(testloader))*100,2),'%',end='\r')
                
                    test_data, test_label,test_name = aud

                    num = ['audio_signals/audio' + name.replace('mbe','').split('.')[0]+'.wav' for name in test_name]
                    with torch.no_grad():
                        pred_i = model.forward(test_data)
                    label = test_label.to('cpu')
                    pred_i = pred_i.to('cpu')
                    pred_t = np.asarray(pred_i >= th)
                    pred_t = pred_t.transpose(0,2,1)
                    df = None if j == 0 else pd.read_csv(filter_file[0]+'_pred/'+ str(th).replace('.','_')+'.csv', sep = ' ')
                    df_t = metric.save_batch_prediction(num,pred_t,categ,df)
                    df_t.to_csv(filter_file[0]+'_pred/'+ str(th).replace('.','_')+'.csv', sep = ' ',index=False)
                    del pred_i
                    del test_data
                    del label
                    torch.cuda.empty_cache()
                i+=1
                print('\n')
            os.makedirs('models/'+filter_file[0])
            shutil.move('models/'+fold_model,'models/'+filter_file[0]+'/'+fold_model)