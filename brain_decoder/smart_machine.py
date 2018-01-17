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
from mymodule.optim import Eve, YFOptimizer
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from load_data import get_data, get_data_multi, get_crops, get_crops_multi
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

epochs = 200
batch_size = 16
cv_splits = 5
torch.manual_seed(1214)
torch.cuda.manual_seed_all(1214)
num_of_subjects = 30



class Conv_lstm(nn.Module):
  def __init__(self):
    super(Conv_lstm, self).__init__()
    self.relu = nn.LeakyReLU()
    self.conv_time = nn.Conv2d(1, 40, (25, 1))
    self.batchnorm1 = nn.BatchNorm2d(40)
    self.conv_spat = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.batchnorm2 = nn.BatchNorm2d(40)
    self.pool = nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1))
    self.dropout = nn.Dropout2d(p=0.5)
    self.lstm = LSTM(40, 10, batch_size, bidirectional=False, num_layers=2,
                     gpu=True, return_seq=False)
    self.dropout_linear = nn.Dropout(p=0.5)
    self.classifier = nn.Linear(10, 2)

  def forward(self, x):
    h = self.conv_time(x)
    h = self.relu(h)
    h = self.batchnorm1(h)
    h = self.conv_spat(h)
    h = self.relu(h)
    h = self.batchnorm2(h)
    h = self.pool(h)
    h = self.dropout(h)
    h = h.squeeze().transpose(1, 2)
    h = self.lstm(h)
    h = self.relu(h)
    h = self.dropout_linear(h)
    h = self.classifier(h)
    return h

class DeepFBCSP(nn.Module):
  def __init__(self):
    super(DeepFBCSP, self).__init__()
    self.conv_time = nn.Conv2d(1, 40, (25, 1))
    self.batchnorm1 = nn.BatchNorm2d(40)
    self.conv_spat1 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.conv_spat2 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.conv_spat3 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.conv_spat4 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.conv_spat5 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.conv_spat6 = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.batchnorm21 = nn.BatchNorm2d(40)
    self.batchnorm22 = nn.BatchNorm2d(40)
    self.batchnorm23 = nn.BatchNorm2d(40)
    self.batchnorm24 = nn.BatchNorm2d(40)
    self.batchnorm25 = nn.BatchNorm2d(40)
    self.batchnorm26 = nn.BatchNorm2d(40)
    self.pool = nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1))
    self.dropout = nn.Dropout2d(p=0.5)
    self.score_ave = nn.AdaptiveAvgPool2d((1, 1))

    self.time_freq_conv = nn.Sequential(nn.Conv2d(6, 100, (12, 5)),
                                     nn.ReLU(),
                                     nn.Dropout2d(p=0.5),
                                     nn.Conv2d(100, 200, (12, 5)),
                                     nn.ReLU(),
                                     nn.Dropout2d(p=0.5),
                                     nn.Conv2d(200, 300, (12, 5)),
                                     nn.ReLU())
    self.dropout_linear = nn.Dropout(p=0.5)
    self.classifier = nn.Linear(300, 2)

  def forward(self, x):
    h = self.conv_time(x)
    h = self.batchnorm1(h)
    h1 = self.conv_spat1(h)
    # h1 = self.batchnorm21(h1)
    h2 = self.conv_spat2(h)
    # h2 = self.batchnorm22(h2)
    h3 = self.conv_spat3(h)
    # h3 = self.batchnorm23(h3)
    h4 = self.conv_spat4(h)
    # h4 = self.batchnorm24(h4)
    h5 = self.conv_spat5(h)
    # h5 = self.batchnorm25(h5)
    h6 = self.conv_spat6(h)
    # h6 = self.batchnorm26(h6)
    h = torch.cat([h1,h2,h3,h4,h5,h6],dim=3)
    h = h.transpose(1,-1)
    h = self.time_freq_conv(h)
    h = self.score_ave(h).view(-1,300)
    h = self.dropout(h)
    h = self.classifier(h)
    return h

model_class = Conv_lstm
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def cv_train(model_class, criterion_class, optimizer_class, X, y,
             epoch=100, num_of_cv=10, batch_size=16):
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
        optimizer = optimizer_class(model.parameters(), lr=1e-4)
        trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer, gpu=True, log=False)
        trainer.run(epochs=epoch)
        accuracy.append(trainer.val_best_acc)
    return accuracy


all_accs_list = []
all_mean_list = []
all_var_list = []

for idx in range(num_of_subjects):
    X, y = get_data(id=idx+1, event_code=[6,10,14], filter=[0.5, 30], t=[0., 4])
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).transpose(0,1,3,2)

#     model = Conv_lstm()
#     model.cuda()

    acc = cv_train(model_class, torch.nn.CrossEntropyLoss,
                   torch.optim.Adam, X, y, epoch=epochs,
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
plt.title('histgram of cross validation accracy')
plt.xlabel('accuracy')
sns.distplot(accs, kde=False);
plt.show()
