import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from codes import utils

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class SEDnet(nn.Module):
    
    def __init__(self,n_class):
        
        super(type(self), self).__init__()
        self.CNN1 = nn.Sequential(
                                nn.Conv2d(1, 128, kernel_size = (3, 3), padding='same'),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 5)),
                                nn.Dropout(p = 0.5, inplace=True))
        self.CNN2 = nn.Sequential(
                                nn.Conv2d(128,128, kernel_size=(3, 3), padding='same'),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2)),
                                nn.Dropout(p=0.5, inplace=True))
        self.CNN3 = nn.Sequential(
                                nn.Conv2d(128,128, kernel_size=(3, 3), padding='same'),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2)),
                                nn.Dropout(p=0.5, inplace=True))
        self.RNN = nn.Sequential(
                                nn.GRU(input_size=256,hidden_size=32,dropout=0.5,bidirectional=True,batch_first=True),
                                utils.SelectItem(0),
                                nn.Dropout(p=0.5),
                                nn.GRU(input_size=64,hidden_size=32,dropout=0.5,bidirectional=True,batch_first=True),
                                utils.SelectItem(0),
                                nn.Dropout(p=0.5))
        self.FC  = nn.Sequential(
                                TimeDistributed(nn.Linear(64, out_features = 32)),
                                nn.Dropout(p=0.5))
        self.output = nn.Sequential(
                                    TimeDistributed(nn.Linear(32, out_features =n_class)))
        
    def forward(self,x):
        
        z = self.CNN1(x)        
        z = self.CNN2(z)        
        z = self.CNN3(z)        
        z = z.permute(0,2,1,3)        
        z = z.reshape((z.shape[0],z.shape[-3], -1))        
        z = self.RNN(z)        
        z = self.FC(z)        
        z = self.output(z)        
        z = torch.sigmoid(z)
        
        return z
    