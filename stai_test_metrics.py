# %%
# Librerias
#%matplotlib notebook
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from codes import metric as mt
from codes.models import STAInet
from codes.dataset2 import WetSoundDataset
import warnings
warnings.filterwarnings("ignore")
# %% [markdown]
# # Test

# %%
main = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# %%
FOLDER_ANNOTATIONS = "datos/"
AUDIO_DIR = os.path.join(main,"features/test_set/stai/")
SAMPLE_RATE = 44100
LEN_SEC = 300
LEN_SAMPLES = LEN_SEC*SAMPLE_RATE
NUM_SAMPLES = 256  #Cantidad de secuencias , no largo de secuencia 12920/40
categories = pd.read_csv(FOLDER_ANNOTATIONS+'categories.txt',names=['class'])
__class_labels = {str(c[0]):i for i,c in enumerate(categories.values)}
categ = {i:str(c[0]) for i,c in enumerate(categories.values)}
N_FFT = 2048
HOP = int(N_FFT/2)
N_MELS = 40
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
thresholds = [0.5]

path_models = "models/40mels/"
results = []
for fold_model in os.listdir(path_models):
    filter_file = fold_model.split(".")
    if len(filter_file)>1:
        if filter_file[1]=="pt":
            print(filter_file[0])
            model = STAInet(NUM_SAMPLES,n_class=len(__class_labels))
            model.load_state_dict(torch.load(path_models+fold_model))
            model.to(device)

            pred = list()
            targets = list()
            tf1 = 0
            ter = 0
            testloader = DataLoader(wet,batch_size=8,shuffle=False)
            ntest = len(testloader)

            for th in thresholds:
                print('Threshold', th, end='\n')
                for j,aud in enumerate(testloader):
                    print(round(((j+1)/len(testloader))*100,2),'%',end='\r')
                
                    tdata, tlabels,tname = aud

                    num = ['audio_signals/audio' + name.replace('mbe','').split('.')[0]+'.wav' for name in tname]
                    with torch.no_grad():
                        toutputs = model.forward(tdata)
                    j_tf1 = mt.f1_score(toutputs,tlabels)
                    j_ter = mt.error_rate(toutputs,tlabels)
                    tf1 += j_tf1
                    ter += j_ter

                tf1 = tf1/ntest
                ter = ter/ntest
                print('F1:{};ER:{}'.format(tf1,ter))
                print('\n')