import os, sys
os.chdir('/home/seigyo/Documents/pytorch/brain_decoder')
sys.path.append(os.pardir)
import numpy as np
from numpy.random import RandomState
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import mne
from mne.io import concatenate_raws
from mymodule.utils import data_loader, evaluator
from mymodule.layers import LSTM, Residual_block, Res_net, Wavelet_cnn, NlayersSeqConvLSTM
from mymodule.trainer import Trainer
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from load_data import get_data, get_data_multi, get_crops, get_crops_multi
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


def elec_map2d(X):
    X_spat = np.zeros((45, X.shape[-1], 1, 5, 7))
    X_spat[:, :, 0, 1, 0] = X[:, 0, :]
    X_spat[:, :, 0, 1, 1] = X[:, 1, :]
    X_spat[:, :, 0, 1, 2] = X[:, 2, :]
    X_spat[:, :, 0, 1, 3] = X[:, 3, :]
    X_spat[:, :, 0, 1, 4] = X[:, 4, :]
    X_spat[:, :, 0, 1, 5] = X[:, 5, :]
    X_spat[:, :, 0, 1, 6] = X[:, 6, :]

    X_spat[:, :, 0, 2, 0] = X[:, 7, :]
    X_spat[:, :, 0, 2, 1] = X[:, 8, :]
    X_spat[:, :, 0, 2, 2] = X[:, 9, :]
    X_spat[:, :, 0, 2, 3] = X[:, 10, :]
    X_spat[:, :, 0, 2, 4] = X[:, 11, :]
    X_spat[:, :, 0, 2, 5] = X[:, 12, :]
    X_spat[:, :, 0, 2, 6] = X[:, 13, :]

    X_spat[:, :, 0, 3, 0] = X[:, 14, :]
    X_spat[:, :, 0, 3, 1] = X[:, 15, :]
    X_spat[:, :, 0, 3, 2] = X[:, 16, :]
    X_spat[:, :, 0, 3, 3] = X[:, 17, :]
    X_spat[:, :, 0, 3, 4] = X[:, 18, :]
    X_spat[:, :, 0, 3, 5] = X[:, 19, :]
    X_spat[:, :, 0, 3, 6] = X[:, 20, :]

    X_spat[:, :, 0, 0, 0] = X[:, 30, :]
    X_spat[:, :, 0, 0, 1] = X[:, 31, :]
    X_spat[:, :, 0, 0, 2] = X[:, 32, :]
    X_spat[:, :, 0, 0, 3] = X[:, 33, :]
    X_spat[:, :, 0, 0, 4] = X[:, 34, :]
    X_spat[:, :, 0, 0, 5] = X[:, 35, :]
    X_spat[:, :, 0, 0, 6] = X[:, 36, :]

    X_spat[:, :, 0, 4, 0] = X[:, 47, :]
    X_spat[:, :, 0, 4, 1] = X[:, 48, :]
    X_spat[:, :, 0, 4, 2] = X[:, 49, :]
    X_spat[:, :, 0, 4, 3] = X[:, 50, :]
    X_spat[:, :, 0, 4, 4] = X[:, 51, :]
    X_spat[:, :, 0, 4, 5] = X[:, 52, :]
    X_spat[:, :, 0, 4, 6] = X[:, 53, :]
    return X_spat

class ConvLSTM(nn.Module):
  def __init__(self):
    super(ConvLSTM, self).__init__()
    self.relu = nn.LeakyReLU()
    self.convlstm = NlayersSeqConvLSTM(input_channels=1,
                        hidden_channels=[16, 16],
                        kernel_sizes=[3,3])
    self.conv = nn.Sequential(nn.Conv2d(16,32,(3,3)),
                      nn.LeakyReLU(),
                      nn.BatchNorm2d(32),
                      nn.Conv2d(32,64,(3,5)),
                      nn.LeakyReLU(),
                      nn.BatchNorm2d(64)).cuda()
    self.dropout = nn.Dropout(0.5)
    self.linear = nn.Linear(64, 2)


  def forward(self, x):
    h, _ = self.convlstm(x)
    h = self.conv(h[:,-1,:,:,:])
    h = h.view(-1, h.size(1))
    h = self.dropout(h)
    h = self.linear(h)
    return h

class Conv3d(nn.Module):
    def __init__(self):
        super(Conv3d, self).__init__()
        self.conv_time = nn.Conv3d(1, 32, (6, 1, 1))
        self.conv_spat = nn.Conv3d(32, 64, (1, 2, 2))
        self.pool_time = nn.AvgPool3d(kernel_size=(40, 1, 1), stride=(10, 1, 1))

        self.conv_time2 = nn.Conv3d(64, 128, (6, 1, 1))
        self.conv_spat2 = nn.Conv3d(128, 256, (1, 2, 2))
        self.pool_time2 = nn.AvgPool3d(kernel_size=(40, 1, 1), stride=(10, 1, 1))
        self.batchnorm = nn.BatchNorm3d(256)
        
        self.adaptivepool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(256, 512)
        self.batchnormlinear = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 2)
    
    def forward(self, x):
        h = self.conv_time(x)
        h = self.conv_spat(h)
        h = self.pool_time(h)
        h = self.conv_time2(h)
        h = self.conv_spat2(h)
        h = self.pool_time2(h)
        h = self.batchnorm(h)
        h = self.adaptivepool(h)
        h = h.squeeze()
        h = self.linear(h)
        h = self.batchnormlinear(h)
        h = self.linear2(h)
        return h

class Conv3d_convLSTM(nn.Module):
    def __init__(self):
        super(Conv3d_convLSTM, self).__init__()
        self.pad = nn.ReplicationPad3d((1,1,1,1,0,0))

        self.conv_time = nn.Conv3d(1, 32, (6, 1, 1))
        self.conv_spat = nn.Conv3d(32, 32, (1, 3, 3))
        self.pool_time = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.batchnorm = nn.BatchNorm3d(32)       

        self.conv_time2 = nn.Conv3d(32, 64, (6, 1, 1))
        self.conv_spat2 = nn.Conv3d(64, 64, (1, 3, 3))
        self.pool_time2 = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.batchnorm2 = nn.BatchNorm3d(64)       

        self.conv_time3 = nn.Conv3d(64, 128, (6, 1, 1))
        self.conv_spat3 = nn.Conv3d(128, 128, (1, 3, 3))
        self.pool_time3 = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))       
        self.batchnorm3 = nn.BatchNorm3d(128)       

        self.convlstm = NlayersSeqConvLSTM(input_channels=128,
                                hidden_channels=[256],
                                kernel_sizes=[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        h = self.conv_time(x)
        h = self.pad(h)
        h = self.conv_spat(h)
        h = self.batchnorm(h)
        h = self.pool_time(h)

        h = self.conv_time2(h)
        h = self.pad(h)
        h = self.conv_spat2(h)
        h = self.batchnorm2(h)
        h = self.pool_time2(h)

        h = self.conv_time3(h)
        h = self.pad(h)
        h = self.conv_spat3(h)
        h = self.batchnorm3(h)
        h = self.pool_time3(h)

        h = h.transpose(1, 2) # (batch, ch, seq, h, w) -> (batch, seq, ch, h, w)
        h, _ = self.convlstm(h)
        h = h[:,-1,:,:,:] # last seq
        h = self.avgpool(h).squeeze()
        h = self.dropout(h)
        return self.linear(h)


class Conv(nn.Module):
  def __init__(self):
    super(Conv, self).__init__()
    self.conv_time = nn.Conv2d(1, 32, (6, 1))
    self.conv_spat = nn.Conv2d(32, 64, (1, 64), bias=False)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.pool = nn.AvgPool2d(kernel_size=(20, 1), stride=(5, 1))
    self.dropout = nn.Dropout2d(p=0.5)
    self.conv_class = nn.Conv2d(64, 2, (12, 1))
    self.score_ave = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    h = self.conv_time(x)
    h = self.conv_spat(h)
    h = self.batchnorm2(h)
    h = self.pool(h)
    h = self.dropout(h)
    h = self.conv_class(h)
    h = self.score_ave(h)
    # print(h.size())
    # return h
    return h.squeeze()

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def cv_train(model_class, criterion_class, optimizer_class, X, y,
             epoch=100, num_of_cv=10, batch_size=16, lr=1e-4):
    kf = KFold(n_splits=num_of_cv, shuffle=False)
    accuracy = []
    for train_idx, val_idx in kf.split(X=X, y=y):
        train_x, val_x = X[train_idx], X[val_idx]
        train_y, val_y = y[train_idx], y[val_idx]
        train_loader = data_loader(train_x, train_y, batch_size=batch_size,
                           shuffle=True, gpu=False)
        val_loader = data_loader(val_x, val_y, batch_size=batch_size)
        writer = SummaryWriter()
        model = model_class().cuda()
        criterion = criterion_class()
        optimizer = optimizer_class(model.parameters(), lr=lr)
        trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer, gpu=True, log=False)
        trainer.run(epochs=epoch)
        accuracy.append(trainer.val_best_acc)
    return accuracy


def main(num_of_subjects, cv_splits, batch_size, epochs, model_class, crop, lr, map2d=False, conv3d=False):

    all_accs_list = []
    all_mean_list = []
    all_var_list = []
    
    for idx in range(num_of_subjects):
        if crop:
            X, y, _, _ = get_crops(id=idx+1, event_code=[4,8,12], filter=[0.5, 30], t=[0., 4],
                                   time_window=1.0, time_step=0.5)
        else:
            X, y = get_data(id=idx+1, event_code=[4,8,12], filter=[0.5, 30], t=[0., 4])

        if map2d:
            X = elec_map2d(X)
            print(X.shape)
            if conv3d:
                X = X.transpose(0,2,1,3,4)
                print(X.shape)
        else:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).transpose(0,1,3,2)

        acc = cv_train(model_class, torch.nn.CrossEntropyLoss,
                    torch.optim.Adam, X, y, epoch=epochs, lr=lr,
                    num_of_cv=cv_splits, batch_size=batch_size)

        mean = np.mean(acc)
        var = np.var(acc)
        print('subject{}   mean_acc:{}, var_acc:{}'.format(idx+1, mean, var))

        all_accs_list.append(acc)
        all_mean_list.append(mean)
        all_var_list.append(var)

    all_mean = np.mean(all_accs_list)
    all_var = np.var(all_accs_list)
    sub_mean = np.mean(all_mean_list)
    sub_var = np.var(all_mean_list)

    print('********************result**********************')
    print('all validation  mean_acc:{}, var_acc:{}'.format(all_mean, all_var))
    print('all subject  mean_acc:{}, var_acc:{}'.format(sub_mean, sub_var))

    # plt.figure(0)
    # accs = np.array(all_accs_list)
    # accs = accs.reshape(-1)
    # sns.set_style("ticks")
    # plt.title('histgram of validation accracy')
    # plt.xlabel('accuracy')
    # sns.distplot(accs, kde=False);
    # plt.show()


    plt.figure(1)
    accs = np.array(all_mean_list)
    accs = accs.reshape(-1)
    sns.set_style("ticks")
    plt.title('histgram of cross validation accracy('+name+')'+taskname)
    plt.xlabel('accuracy')
    sns.distplot(accs, kde=False);
    plt.show()


if __name__ == '__main__':
    epochs = 200
    batch_size = 16
    cv_splits = 5
    torch.manual_seed(1214)
    torch.cuda.manual_seed_all(1214)
    num_of_subjects = 30
    crop = False
    lr = 1e-6
    map2d = False
    conv3d = False
    model = Conv
    taskname = 'right vs left'
    main(num_of_subjects=num_of_subjects, 
         cv_splits=cv_splits, 
         batch_size=batch_size,
         epochs=epochs, 
         model_class=model, 
         crop=crop, 
         lr=lr, 
         map2d=map2d, 
         conv3d=conv3d,
         name='Conv',taskname=taskname)