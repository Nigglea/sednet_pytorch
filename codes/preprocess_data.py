import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torchaudio
from codes import utils

def preprocess_data(audio_sample_path,annotations_file,target_sample_rate,
                                len_samples,N_FFT,HOP,N_MELS,labels,device="cuda"):
    #print(audio_sample_path)
    signal, sr = torchaudio.load(audio_sample_path)

    signal = signal.to(device)
    signal = resample_if_necessary(signal, sr,target_sample_rate)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal,len_samples)
    signal = right_pad_if_necessary(signal,len_samples)
    signal = transformation(signal,target_sample_rate,N_FFT,
                                HOP,N_MELS,trans="logmel")

    label = get_audio_sample_label(signal,target_sample_rate,HOP,
                                audio_sample_path,annotations_file,
                                labels,device="cuda")

    signal = signal.permute(0,2,1)
    label = label.permute(1,0)
    return signal,label

def cut_if_necessary(signal,len_samples):
    if signal.shape[1] > len_samples:
        signal = signal[:, :len_samples]
    return signal

def right_pad_if_necessary(signal,len_samples):
    length_signal = signal.shape[1]
    if length_signal < len_samples:
        num_missing_samples = len_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def resample_if_necessary(signal, sr,target_sample_rate):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def transformation(signal,SAMPLE_RATE,N_FFT,HOP,N_MELS,trans="logmel",device="cuda"):
    if trans == "mel":
        transf = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP ,
        n_mels=N_MELS
        )
        transf = transf.to(device)
        signal = transf(signal)
    if trans == "logmel":
        transf = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP ,
        n_mels=N_MELS,
        normalized = True
        )
        transf = transf.to(device)
        signal = transf(signal)
        signal = torch.log(signal+1e-3)
    return signal

def get_audio_sample_label(signal,target_sample_rate,hop, audio_sample_path,annotations_file,labels,device="cuda"):
    label = torch.zeros((len(labels), signal.shape[2]))
    filenames = utils.load_desc_file(annotations_file,labels)
    tmp_data = np.array(filenames[audio_sample_path])
    frame_start = np.floor(tmp_data[:, 0] * target_sample_rate / hop).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * target_sample_rate / hop).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[val, frame_start[ind]:frame_end[ind]] = 1
    label_tensor = torch.tensor(label,device=device)
    return label_tensor
