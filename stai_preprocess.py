
import os
import numpy as np
import pandas as pd
from codes.acusidxs import *
import time
from codes import utils

main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


FOLDER_ANNOTATIONS = "datos/"
ANNOTATIONS_FILE = ["train_set.txt","test_set.txt"]
N_FOLDS = 4
MBE_DIR_TR = "features/"+ ANNOTATIONS_FILE[0].split(".")[0]+"/stai/"
MBE_DIR_TE = "features/"+ ANNOTATIONS_FILE[1].split(".")[0]+"/stai/"
SAMPLE_RATE = 44100
LEN_SEC = 300
LEN_SAMPLES = LEN_SEC*SAMPLE_RATE
categories = pd.read_csv(FOLDER_ANNOTATIONS+'categories.txt',names=['class'])
__class_labels = {str(c[0]):i for i,c in enumerate(categories.values)}


N_FFT = 2048
HOP = int(N_FFT/2)
N_MELS = 40



df_train = pd.read_csv(FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],sep=" ",
                    names=["filepath","start","end","class_name"])
audio_filenames_train = np.unique(df_train.filepath)


df_test = pd.read_csv(FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],sep=" ",
                    names=["filepath","start","end","class_name"])
audio_filenames_test = np.unique(df_test.filepath)

print('Train:')
for i,file in enumerate(audio_filenames_train):
    stai,label = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    mean_i,std_i = utils.return_mean_std(stai)
    if i == 0:
        global_mean = mean_i
        global_std = std_i
        print(type(global_mean))
    global_mean,global_std = utils.new_mean_std(global_mean,global_std,mean_i,std_i)
    namembe = "{}stai{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":stai,"label":label}, namembe)
    print(i)

print('Test:')
for i,file in enumerate(audio_filenames_test):
    stai,label = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    mean_i,std_i = utils.return_mean_std(stai)
    global_mean,global_std = utils.new_mean_std(global_mean,global_std,mean_i,std_i)
    namembe = "{}stai{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":stai,"label":label}, namembe)
    print(i)
    
print('NORMALIZE')

print('Train:')
for i,file in enumerate(audio_filenames_train):
    stai,label = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    stai = utils.apply_mean_std(stai,global_mean,global_std)
    namembe = "{}stai{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":stai,"label":label}, namembe)
    print(i)

print('Test:')
for i,file in enumerate(audio_filenames_test):
    stai,label = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    stai = utils.apply_mean_std(stai,global_mean,global_std)
    namembe = "{}stai{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"data":stai,"label":label}, namembe)
    print(i)

