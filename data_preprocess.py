
import os
import numpy as np
import pandas as pd
from codes.acusidxs import *
from codes.preprocess_data import preprocess_data
from codes import utils
import time


main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


FOLDER_ANNOTATIONS = "datos/"
ANNOTATIONS_FILE = ["train_set.txt","test_set.txt"]
N_FOLDS = 4
MBE_DIR_TR = "features/"+ ANNOTATIONS_FILE[0].split(".")[0]+"/data/"
MBE_DIR_TE = "features/"+ ANNOTATIONS_FILE[1].split(".")[0]+"/data/"
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
    stai,_ = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    mbe,label = preprocess_data(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    mean_i,std_i = utils.return_mean_std(stai)
    if i == 0:
        global_mean = mean_i
        global_std = std_i
        print(type(global_mean))
    global_mean,global_std = utils.new_mean_std(global_mean,global_std,mean_i,std_i)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    if torch.isnan(mbe).any():
        print(file,'mbe')
    if torch.isnan(stai).any():
        print(file,'stai')
    namembe = "{}data{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"stai":stai,"mbe":mbe,"label":label}, namembe)
    if i==0:
        mean_std = mbe
    else:
        mean_std = torch.cat((mean_std,mbe))

print('Test:')
for i,file in enumerate(audio_filenames_test):
    stai,_ = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    mbe,label = preprocess_data(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    mean_i,std_i = utils.return_mean_std(stai)
    global_mean,global_std = utils.new_mean_std(global_mean,global_std,mean_i,std_i)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    if torch.isnan(mbe).any():
        print(file,'mbe')
    if torch.isnan(stai).any():
        print(file,'stai')
    namembe = "{}data{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"stai":stai,"mbe":mbe,"label":label}, namembe)
    mean_std = torch.cat((mean_std,mbe))
    

mean_mbe = torch.mean(mean_std)
std_mbe = torch.std(mean_std)
print('MBE:',mean_mbe,std_mbe)


print('Train:')
for i,file in enumerate(audio_filenames_train):
    stai,_ = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    mbe,label = preprocess_data(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[0],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")

    #stai = utils.apply_mean_std(stai,global_mean,global_std)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    if torch.isnan(mbe).any():
        print(file,'mbe')
    if torch.isnan(stai).any():
        print(file,'stai')
    namembe = "{}data{}.pt".format(MBE_DIR_TR,numberfile)
    namembe = os.path.join(main,namembe)
    torch.save({"stai":stai,"mbe":mbe,"label":label}, namembe)
    mbe = (mbe-mean_mbe)/std_mbe
    if i==0:
        mean_std = mbe
    else:
        mean_std = torch.cat((mean_std,mbe))

print('Test:')
for i,file in enumerate(audio_filenames_test):
    stai,_ = st_ai(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,__class_labels,device="cuda")
    stai = stai.to(dtype=torch.float32)
    mbe,label = preprocess_data(file,main,FOLDER_ANNOTATIONS+ANNOTATIONS_FILE[1],SAMPLE_RATE,
                                LEN_SAMPLES,N_FFT,HOP,N_MELS,__class_labels,device="cuda")
    #stai = utils.apply_mean_std(stai,global_mean,global_std)
    numberfile = file.split("/")[1].split(".")[0].replace("audio","")
    if torch.isnan(mbe).any():
        print(file,'mbe')
    if torch.isnan(stai).any():
        print(file,'stai')
    namembe = "{}data{}.pt".format(MBE_DIR_TE,numberfile)
    namembe = os.path.join(main,namembe)
    mbe = (mbe-mean_mbe)/std_mbe
    torch.save({"stai":stai,"mbe":mbe,"label":label}, namembe)
    mean_std = torch.cat((mean_std,mbe))

mean_mbe = torch.mean(mean_std)
std_mbe = torch.std(mean_std)
print('MBE:',mean_mbe,std_mbe)
