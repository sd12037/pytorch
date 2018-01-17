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
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.util import set_random_seeds
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
from mymodule.utils import data_loader, evaluator
from mymodule.layers import LSTM, Residual_block, Res_net
from mymodule.trainer import Trainer
from mymodule.optim import Eve, YFOptimizer
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from load_data import get_data, get_data_multi, get_crops, get_crops_multi


'''
sub_id can be range of 1-109.
runs
1: baseline of eyes open
2: baseline of eyes close
3,7,11: motor execution left vs right hands
4,8,12: motor imagery left vs right hands
5,9,13: motor execution hands vs feet
6,10,14: motor imagery hands vs feet
'''

# X, y = get_data(id=1, event_code=[5,6,9,10,13,14], filter=[5, 25], t=[-1., 4])
# train_X, train_y = X[:60], y[:60]
# test_X, test_y = X[60:], y[60:]
#
# # train_X, train_y = get_data_multi(sub_id_range=[1, 51], event_code=[5,9,13], filter=[0.5,36], t=[0.5, 4.0])
# # test_X, test_y = get_data_multi(sub_id_range=[51, 55], event_code=[5,9,13], filter=[0.5,36], t=[0.5, 4.0])
#
# f_dim = train_X.shape[1]
# seq_len = train_X.shape[2]
#
# len(train_X)

# X_crop, y_crop = get_crops(id=1, event_code=[6,10,14], filter=[0.5, 36],
#                            t=[0, 4.0], time_window=1.0, time_step=1/160)
X_crop, y_crop = get_crops_multi(sub_id_range=[1, 20], event_code=[6,10,14],
                                 t=[0, 4.0], filter=[0.5,36],
                                 time_window=1.0, time_step=0.25)
print(X_crop.shape)

cut = int(X_crop.shape[0]*0.8)
train_X = X_crop[:cut].reshape(-1, 1, X_crop.shape[-1], X_crop.shape[1])
test_X = X_crop[cut:].reshape(-1, 1, X_crop.shape[-1], X_crop.shape[1])
train_y = y_crop[:cut]
test_y = y_crop[cut:]


'''
sub_id can be range of 1-109.
runs
1: baseline of eyes open
2: baseline of eyes close
3,7,11: motor execution left vs right hands
4,8,12: motor imagery left vs right hands
5,9,13: motor execution hands vs feet
6,10,14: motor imagery hands vs feet
'''
#
# X, y = get_data(id=1, event_code=[5,9,13], filter=None, t=[-0.5, 4])
# X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).transpose(0,1,3,2)
# train_X, train_y = X[:30], y[:30]
# test_X, test_y = X[30:], y[30:]

# train_X, train_y = get_data_multi(sub_id_range=[1, 50], event_code=[6,10,14], filter=None, t=[0.5, 4.0])
# train_X = train_X.reshape(train_X.shape[0],
#                           1, train_X.shape[1],
#                           train_X.shape[2]).transpose(0,1,3,2)
# test_X, test_y = get_data_multi(sub_id_range=[50, 55], event_code=[6,10,14], filter=None, t=[0.5, 4.0])
# test_X = test_X.reshape(test_X.shape[0],
#                         1, test_X.shape[1],
#                         test_X.shape[2]).transpose(0,1,3,2)

f_dim = train_X.shape[3]
seq_len = train_X.shape[2]
'''
モデル
'''
epochs = 200
batch_size = 512
lr = 1e-6
train_loader = data_loader(train_X, train_y, batch_size=batch_size,
                           shuffle=True, gpu=False)
# test_loader = data_loader(test_, test_t, batch_size=batch_size)
val_loader = data_loader(test_X, test_y, batch_size=batch_size)

class Conv_lstm(nn.Module):
  def __init__(self):
    super(Conv_lstm, self).__init__()
    self.conv_time = nn.Conv2d(1, 40, (24, 1))
    self.conv_spat = nn.Conv2d(40, 40, (1, 64), bias=False)
    self.batchnorm = nn.BatchNorm2d(40)
    self.pool = nn.AvgPool2d(kernel_size=(80, 1), stride=(20, 1))
    self.dropout = nn.Dropout2d(p=0.5)
    self.lstm = LSTM(40, 10, batch_size, gpu=True, return_seq=False)
    self.classifier = nn.Linear(10, 2)

  def forward(self, x):
    h = self.conv_time(x)
    h = self.conv_spat(h)
    h = self.batchnorm(h)
    h = self.pool(h)
    h = self.dropout(h)
    h = h.squeeze().transpose(1, 2)
    h = self.lstm(h)
    h = self.classifier(h)
    return h

model = Conv_lstm()
model.cuda()
#
# y = model(Variable(torch.from_numpy(train_X)).cuda())
# print(y.size())


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()

### train_loop
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer, gpu=True)
trainer.run(epochs=epochs)
