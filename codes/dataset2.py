import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torchaudio
from codes import utils

class WetSoundDataset(Dataset):

    def __init__(self,
                 mbe_dir,
                 len_mbe,
                 num_samples,
                 class_labels,
                 device):
        self.mbe_dir = mbe_dir
        self.mbe_list = os.listdir(self.mbe_dir)
        self.mbe_len = len_mbe
        self.labels = class_labels
        self.device = device
        self.num_samples = num_samples

    def __len__(self):
        self.len_set = self.mbe_len*len(self.mbe_list)
        return self.len_set

    def __getitem__(self, index):
        index_path = int(index // (self.mbe_len))
        index_samples = int(index - index_path*self.mbe_len)
        mbe_sample_path = self.mbe_list[index_path]
        mbe = torch.load(self.mbe_dir+mbe_sample_path)
        signal = mbe["data"]
        label =mbe["label"]
        begin_sec = (index_samples*self.num_samples)
        end_sec = ((index_samples+1)*self.num_samples)
        #print(begin_sec,end_sec)
        signal = signal[:,begin_sec:end_sec,:]
        label = label[begin_sec:end_sec,:]
        return signal, label 
