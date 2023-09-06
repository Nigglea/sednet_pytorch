# %%
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torchaudio
import codes.utils
from codes.preprocess_data import *

# %%
main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(main)
maina = os.path.join(main,'TUT-sound-events-2017-development')
# %%
FOLDER_ANNOTATIONS = "datos/"
ANNOTATIONS_FILE = ["tut__train_set.txt","tut__test_set.txt"]
N_FOLDS = 4
MBE_DIR_TR = "features/"+ ANNOTATIONS_FILE[0].split(".")[0].split('__')[1]+"/tut/"
MBE_DIR_TE = "features/"+ ANNOTATIONS_FILE[1].split(".")[0].split('__')[1]+"/tut/"
SAMPLE_RATE = 44100
LEN_SEC = 300
LEN_SAMPLES = LEN_SEC*SAMPLE_RATE
categories = pd.read_csv('../TUT-sound-events-2017-development/tut_categories.txt',names=['class'])
__class_labels = {str(c[0]):i for i,c in enumerate(categories.values)}
N_FFT = 2048
HOP = int(N_FFT/2)
N_MELS = 40


df_train = pd.read_csv(FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],sep=" ",
                    names=["filepath","start","end","class_name"])
audio_filenames_train = np.unique(df_train.filepath)
print(audio_filenames_train)

df_test = pd.read_csv(FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],sep=" ",
                    names=["filepath","start","end","class_name"])
audio_filenames_test = np.unique(df_test.filepath)
print(audio_filenames_test)

for i,file in enumerate(audio_filenames_train):
    mbe,label = preprocess_data(file,maina,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    numberfile = file.split("/")[2].split(".")[0].replace("audio","")
    namembe = "{}mbe{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":mbe,"label":label}, namembe)
    if i==0:
        mean_std = mbe
    else:
        mean_std = torch.cat((mean_std,mbe))
    torch.cuda.empty_cache()


for i,file in enumerate(audio_filenames_test):
    mbe,label = preprocess_data(file,maina,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    namembe = "{}mbe{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":mbe,"label":label}, namembe)
    mean_std = torch.cat((mean_std,mbe))


mean = torch.mean(mean_std)
std = torch.std(mean_std)
print(mean,std)


for i,file in enumerate(audio_filenames_train):
    mbe,label = preprocess_data(file,maina,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    namembe = "{}mbe{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    mbe = (mbe-mean)/std
    torch.save({"data":mbe,"label":label}, namembe)
    if i==0:
        mean_std = mbe
    else:
        mean_std = torch.cat((mean_std,mbe))

for i,file in enumerate(audio_filenames_test):

    mbe,label = preprocess_data(file,maina,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    namembe = "{}mbe{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    mbe = (mbe-mean)/std
    torch.save({"data":mbe,"label":label}, namembe)
    mean_std = torch.cat((mean_std,mbe))

mean1 = torch.mean(mean_std)
std1 = torch.std(mean_std)
print(mean1,std1)