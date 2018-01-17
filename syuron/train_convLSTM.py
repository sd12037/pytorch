import os, sys
sys.path.append(os.pardir)
import numpy as np
from numpy.random import RandomState
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import make_4class, make_4class_crops
from data_loader import make_class, make_class_crop
from mymodule.utils import data_loader, num_of_correct, evaluator
from data_arg import time_normalization, elec_map2d
from trainer import Trainer
from mymodule.layers import NlayersSeqConvLSTM


def normal_data(sub_range=[1,10], bpfilter=[0.5, 45], problem='lr'):
    if problem == '4':
        X, y = make_4class(subject_id=sub_range,
                        bpfilter = bpfilter)
    else:
        X, y = make_class(subject_id=sub_range,
                        problem=problem,
                        bpfilter = bpfilter)
    return X, y

def crop_data(sub_range=[1,10], bpfilter=[0.5, 45], problem='lr',
              time_window=1, time_step=0.5):
    if problem == '4':
        X, y = make_4class_crops(subject_id=sub_range,
                                bpfilter = bpfilter,
                                time_window=time_window,
                                time_step=time_step)
    else:
        X, y = make_class_crop(subject_id=sub_range,
                            problem=problem,
                            bpfilter = bpfilter,
                            time_window=time_window,
                            time_step=time_step)
    return X, y

def split_data(X, y, split_rate=0.8):
    X, y = shuffle(X, y)
    X_train, y_train = X[:int(X.shape[0]*split_rate)], y[:int(y.shape[0]*split_rate)]
    X_val, y_val = X[int(X.shape[0]*split_rate):], y[int(y.shape[0]*split_rate):]
    return X_train, y_train, X_val, y_val

class Conv3d_convLSTM(nn.Module):
    def __init__(self):
        super(Conv3d_convLSTM, self).__init__()
        self.pad = nn.ReplicationPad3d((1,1,1,1,0,0))

        self.conv_time = nn.Conv3d(1, 8, (25, 1, 1))
        self.conv_spat = nn.Conv3d(8, 8, (1, 3, 3))
        self.batchnorm = nn.BatchNorm3d(8)       
        self.dropout = nn.Dropout3d(0.2)

        self.conv_time2 = nn.Conv3d(8, 16, (25, 1, 1))
        self.conv_spat2 = nn.Conv3d(16, 16, (1, 3, 3))
        self.batchnorm2 = nn.BatchNorm3d(16)       
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv_time3 = nn.Conv3d(16, 32, (25, 1, 1))
        self.conv_spat3 = nn.Conv3d(32, 32, (1, 3, 3))
        self.pool_time3 = nn.AvgPool3d(kernel_size=(75, 1, 1), stride=(25, 1, 1))       
        self.batchnorm3 = nn.BatchNorm3d(32)       
        self.dropout3 = nn.Dropout3d(0.2)

        self.convlstm = NlayersSeqConvLSTM(input_channels=32,
                                           hidden_channels=[64, 64],
                                           kernel_sizes=[3, 3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)
        self.linear = nn.Linear(64, 4)

    def forward(self, x):
        h = self.conv_time(x)
        h = self.pad(h)
        h = self.conv_spat(h)
        h = self.batchnorm(h)
        h = self.dropout(h)

        h = self.conv_time2(h)
        h = self.pad(h)
        h = self.conv_spat2(h)
        h = self.batchnorm2(h)
        h = self.dropout2(h)

        h = self.conv_time3(h)
        h = self.pad(h)
        h = self.conv_spat3(h)
        h = self.batchnorm3(h)
        # h = self.pool_time3(h)
        h = self.dropout3(h)

        h = h.transpose(1, 2) # (batch, ch, seq, h, w) -> (batch, seq, ch, h, w)
        h, _ = self.convlstm(h)
        h = h[:,-1,:,:,:] # last seq
        h = self.avgpool(h).squeeze()
        h = self.batchnorm4(h)
        h = self.dropout4(h)
        return self.linear(h)



def main(name):
    if name == None:
        name = input("please give name to the training model:")
    else:
        name = name
    batch_size = 128
    # X, y = normal_data(sub_range=[1,50], bpfilter=[0.5, 45], problem='4')
    X, y = crop_data(sub_range=[1,50], bpfilter=[0.5, 45], problem='lr',
                     time_window=1.0, time_step=0.25)
    X = elec_map2d(X)
    X = X.transpose(0,2,1,3,4)
    X_train, y_train, X_val, y_val = split_data(X, y, split_rate=0.8)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    train_loader = data_loader(x=X_train,
                            t=y_train,
                            batch_size=batch_size,
                            shuffle=True,
                            gpu=False)
    val_loader = data_loader(x=X_val,
                             t=y_val,
                             batch_size=batch_size,
                             shuffle=False,
                             gpu=False) 
    model = Conv3d_convLSTM()
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, criterion, optimizer,
                train_loader, val_loader,
                name=name, gpu=True, )
    trainer.run(epochs=500)

if __name__ == '__main__':
    main(None)


