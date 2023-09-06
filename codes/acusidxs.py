import os
import numpy as np
import torch
import torchaudio
from codes import preprocess_data
from scipy import signal as sig
from scipy import fftpack

EPS = 1e-7

def st_ai(audio_sample_path,
          all_path,
          annotations_file,
          target_sample_rate,
          len_samples,
          N_FFT,
          HOP,
          labels,
          device="cuda"):
    """Short-time acoustics indices (st_ai)

    Args:
        audio_sample_path (str): _description_
        all_path (str): _description_
        annotations_file (str): _description_
        target_sample_rate (int): _description_
        len_samples (int): _description_
        N_FFT (int): _description_
        HOP (int): _description_
        labels (dict): _description_
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        tuple: tensor with acoustics indexs and tensor with one hot encoding of labels
    """

    signal, sr = torchaudio.load(os.path.join(all_path, audio_sample_path))

    signal = signal.to(device)
    signal = preprocess_data.normalization(signal, mode="zscore")
    signal = preprocess_data.resample_if_necessary(signal, sr,
                                                   target_sample_rate)
    signal = preprocess_data.mix_down_if_necessary(signal)
    signal = preprocess_data.cut_if_necessary(signal, len_samples)
    signal = preprocess_data.right_pad_if_necessary(signal, len_samples)
    signal = compute_acusidxs(signal, target_sample_rate, N_FFT, HOP, device)

    signal = signal.reshape([1,signal.shape[0],signal.shape[1]])

    label = preprocess_data.get_audio_sample_label(signal, target_sample_rate,
                                                   HOP, audio_sample_path,
                                                   annotations_file, labels,
                                                   device)

    signal = signal.permute(0, 2, 1)
    label = label.permute(1, 0)
    return signal, label


def compute_acusidxs(signal, target_sample_rate, N_FFT, HOP, device='cuda'):
    """
    ACI: Acoustic Complex Index
    BIO: Bioacoustic Index
    NDSI: Normalize Difference Soundscape Index
    H_s: Spectral Entropy
    H_t: Temporal Entropy
    M: Median of Envelope Signal

    Args:
        signal (_type_): _description_
        target_sample_rate (_type_): _description_
        N_FFT (_type_): _description_
        HOP (_type_): _description_
        device (str, optional): _description_. Defaults to 'cuda'.

    Returns:
        _type_: _description_
    """

    signal = filterband(signal, target_sample_rate, device)

    spectro, frequencies = compute_specgram(signal,
                                            target_sample_rate,
                                            N_FFT,
                                            HOP,
                                            square=False,
                                            device=device)
    frames, _ = compute_wavegram(signal, N_FFT, HOP, device=device)
    frames = frames.cpu().detach().numpy()

    ACI = compute_aci(signal, target_sample_rate, N_FFT, HOP, device)
    #ACI = (ACI - torch.min(ACI)) / (torch.max(ACI) - torch.min(ACI))
    

    BIO = compute_bio(spectro = spectro, frequencies = frequencies, device=device)
    #BIO = (BIO - torch.min(BIO)) / (torch.max(BIO) - torch.min(BIO))

    NDSI = compute_ndsi(frames, target_sample_rate, device = device)
    #NDSI = (NDSI - torch.min(NDSI)) / (torch.max(NDSI) - torch.min(NDSI))
    

    SH = compute_sh(spectro, device)
    #SH = (SH - torch.min(SH)) / (torch.max(SH) - torch.min(SH))
    

    TH = compute_th(frames, device)
    #TH = (TH - torch.min(TH)) / (torch.max(TH) - torch.min(TH))
    

    M = compute_m(frames, device)
    #M = (M - torch.min(M)) / (torch.max(M) - torch.min(M))
    

    idxs = torch.stack([ACI, BIO, NDSI, SH, TH, M])

    return idxs


def filterband(signal, target_sample_rate, device='cpu'):
    """Butterworth highpass filter

    Args:
        signal (tensor): signal to filter
        target_sample_rate (int): sample rate
        device (str): Device where is compute.

    Returns:
        tensor: signal filtered
    """

    freq_filter = 300
    order = 8
    Wn = freq_filter / float(target_sample_rate / 2)

    [b, a] = sig.butter(order, Wn, btype='highpass')

    sig_f = sig.filtfilt(b, a, signal.cpu().detach().numpy())
    sig_f = torch.Tensor(sig_f.copy()).to(device)
    return sig_f


def compute_specgram(signal,
                     target_sample_rate,
                     N_FFT,
                     HOP,
                     device='cpu',
                     square=True,
                     padding=False):
    """_summary_

    Args:
        signal (_type_): _description_
        target_sample_rate (_type_): _description_
        N_FFT (_type_): _description_
        HOP (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        square (bool, optional): _description_. Defaults to True.
        padding (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if len(signal.shape)>1:
        signal=signal.squeeze()

    sig_d = signal.to(device=device)
    if padding:
        l_pad = N_FFT - HOP
        l_zeros = torch.zeros(l_pad, device=device)
        sig_d = torch.cat((l_zeros, sig_d))
    if square:
        p = 2.0
    else:
        p = 1.0

    spec = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP,
        center=True,
        pad_mode="reflect",
        power=p,
    )

    spec.to(device)

    spectro = spec(sig_d)

    frequencies = torch.linspace(0,
                                 target_sample_rate // 2,
                                 steps=spectro.shape[0],
                                 device=device)
    return spectro, frequencies


def compute_wavegram(signal, N_FFT, HOP, device, centered=False):
    """_summary_

    Args:
        signal (_type_): _description_
        N_FFT (_type_): _description_
        HOP (_type_): _description_
        device (_type_): _description_
        centered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if len(signal.shape)>1:
        signal = signal.squeeze()


    if len(signal) % HOP != 0:
        pad = N_FFT - len(signal) % HOP
        zero_pad = torch.zeros((pad)).to(device)
        signal = torch.cat((signal, zero_pad),dim=0)

    W = sig.get_window('hann', N_FFT, fftbins=False)

    if centered:
        time_shift = int(N_FFT / 2)
        times = range(time_shift,
                      len(signal) + 1 - time_shift, HOP)  # centered
        print("centered", times[0], times[-1])
        frames = [signal[i - time_shift:i + time_shift] * W
                  for i in times]  # centered frames
        framest = signal.unfold(0, N_FFT, HOP)
        print(torch.stack(frames), framest)
    else:
        times = torch.linspace(0,
                               len(signal) - N_FFT + 1,
                               (len(signal) - N_FFT + 1) // HOP).to(device)
        frames = np.multiply(signal.unfold(0, N_FFT, HOP).to('cpu'),W)

    return frames, times


def compute_aci(signal, target_sample_rate, N_FFT, HOP, device='cpu'):
    """_summary_

    Args:
        signal (_type_): _description_
        target_sample_rate (_type_): _description_
        N_FFT (_type_): _description_
        HOP (_type_): _description_

    Returns:
        _type_: _description_
    """

    spec_aci, _ = compute_specgram(signal,
                                   target_sample_rate,
                                   N_FFT,
                                   HOP,
                                   square=False,
                                   padding=True,
                                   device=device)

    t_spectro = spec_aci
    t_spec_diff = torch.abs(torch.diff(t_spectro))
    t_spec_sum = torch.sum(t_spectro, dim=0)
    t_spec_aci = t_spec_diff / t_spec_sum[1:]
    t_aci = torch.sum(t_spec_aci, dim=0)
    t_aci = t_aci / torch.max(t_aci)
    return t_aci


def compute_bio(spectro,
                frequencies,
                min_freq=2000,
                max_freq=8000,
                device='cpu'):
    """_summary_

    Args:
        spectro (_type_): _description_
        frequencies (_type_): _description_
        min_freq (int, optional): _description_. Defaults to 2000.
        max_freq (int, optional): _description_. Defaults to 8000.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """

    if device == 'cpu':
        t_frequencies = frequencies
        t_min = torch.Tensor([min_freq], device=device)
        t_max = torch.Tensor([max_freq], device=device)
        t_spectro = spectro
    elif device == 'cuda':
        t_frequencies = frequencies.to(device)
        t_min = torch.Tensor([min_freq]).to(device)
        t_max = torch.Tensor([max_freq]).to(device)
        t_spectro = spectro.to(device)

    min_f_bin = torch.argmin(torch.abs(t_frequencies - t_min))
    max_f_bin = torch.ceil(
        torch.argmin(torch.abs(t_frequencies - t_max)).float()).int()
    min_f_bin = min_f_bin - 1

    t_spectro_BI = 20 * torch.log10(t_spectro / torch.max(t_spectro))
    t_spectre_BI_segment = t_spectro_BI[min_f_bin:max_f_bin, :]
    t_spectre_BI_segment_normalized = t_spectre_BI_segment - torch.min(
        t_spectre_BI_segment, dim=0)[0]
    t_bin_area = t_spectre_BI_segment_normalized / (t_frequencies[1] -
                                                    t_frequencies[0])
    t_area_temporal = torch.sum(t_bin_area, dim=0)
    t_area_temporal = t_area_temporal / torch.max(t_area_temporal)
    return t_area_temporal


def compute_ndsi(frames,
                 target_sample_rate,
                 anthrophony=[1000, 2000],
                 biophony=[2000, 11000],
                 device='cpu'):
    """_summary_

    Args:
        frames (_type_): _description_
        target_sample_rate (_type_): _description_
        anthrophony (list, optional): _description_. Defaults to [1000, 2000].
        biophony (list, optional): _description_. Defaults to [2000, 11000].
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """

    welchs_f, welchs_p = sig.welch(frames,
                                   fs=target_sample_rate,
                                   window='hamming',
                                   detrend='constant',
                                   return_onesided=True,
                                   scaling='density',
                                   axis=1)
    if device == 'cpu':
        welchs_pow = torch.Tensor(welchs_p * welchs_f[1], device=device)
    elif device == 'cuda':
        welchs_pow = torch.Tensor(welchs_p * welchs_f[1]).to(device)

    
    min_ant_bin = torch.argmin(
        torch.abs(torch.Tensor(welchs_f - anthrophony[0])))
    max_ant_bin = torch.argmin(
        torch.abs(torch.Tensor(welchs_f - anthrophony[1])))

    min_bi_bin = torch.argmin(torch.abs(torch.Tensor(welchs_f - biophony[0])))
    max_bi_bin = torch.argmin(torch.abs(torch.Tensor(welchs_f - biophony[1])))

    ant = torch.sum(welchs_pow[:, min_ant_bin:max_ant_bin], dim=1)
    bi = torch.sum(welchs_pow[:, min_bi_bin:max_bi_bin], dim=1)

    NDSI = torch.divide(bi - ant, bi + ant)
    return NDSI


def compute_sh(spectro, device='cpu'):
    """_summary_

    Args:
        spectro (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """

    spectro = spectro.to(device)
    N = torch.Tensor([spectro.shape[0]]).to(device)
    norm = torch.sum(spectro, dim=0)
    spectro = torch.div(spectro, norm+EPS)
    log_N = torch.log2(N)
    temp_num = -torch.sum(torch.multiply(spectro, torch.log2(spectro+EPS)), dim=0)
    temp_den = (torch.sum(spectro, dim=0) * log_N)+EPS
    temp = torch.divide(temp_num,temp_den)
    return temp


def compute_th(frames, device='cpu'):
    """_summary_

    Args:
        frames (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """
    N = len(frames[0])
    framess = torch.from_numpy(
        sig.hilbert(frames, fftpack.helper.next_fast_len(N))).to(device)
    sad = torch.abs(framess)
    sad_sd = torch.divide(sad.T, torch.sum(sad, axis=1))
    sad_sd = sad_sd.T
    log_N = torch.log2(torch.Tensor([sad_sd.shape[1]]).to(device))
    s = torch.multiply(sad_sd, torch.log2(sad_sd+EPS))
    ss = torch.sum(s, axis=1)
    sad_th = torch.div(-ss, (torch.sum(sad_sd, dim=1) * log_N)+EPS)
    return sad_th


def compute_m(frames, device='cpu'):
    """_summary_

    Args:
        frames (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """
    m_env = torch.abs(
        torch.from_numpy(
            sig.hilbert(frames, fftpack.helper.next_fast_len(len(
                frames[0])))).to(device))

    mm_env = torch.median(m_env, dim=1).values * torch.tensor(4.2950e+09)
    m_env_norm = torch.divide(mm_env, torch.max(mm_env))
    return m_env_norm
