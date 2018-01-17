import os, sys
os.chdir('/home/seigyo/Documents/pytorch/brain_decoder')
import os, sys
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
from torch_sample.torchsample.modules.module_trainer import ModuleTrainer
from torch_sample.torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torch_sample.torchsample.regularizers import L1Regularizer, L2Regularizer
from torch_sample.torchsample.constraints import UnitNorm
from torch_sample.torchsample.metrics import CategoricalAccuracy
from load_data import get_data, get_data_multi, get_crops


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

X_crop, y_crop = get_crops(id=1, event_code=[5,6,9,10,13,14], filter=[5, 25],
                           t=[0, 4], time_window=1.0)

X_crop.shape
y_crop.shape


'''
モデル
'''
epochs = 200
seq_len = X_crop.shape[-1]
batch_size = 1024
lr = 1e-5
train_loader = data_loader(X_crop[:40000], y_crop[:40000], batch_size=batch_size,
                           shuffle=True, gpu=False)
# test_loader = data_loader(test_, test_t, batch_size=batch_size)
val_loader = data_loader(X_crop[40000:], y_crop[40000:], batch_size=batch_size)

### resnet
res_ch = [64, 128]
pooling = [int(seq_len/2), int(seq_len/4)]
res_dropout = 0.4

### lstm
lstm_units = [res_ch[-1], 16]
lstm_dropout = 0.4
bi = True

### linear
dense_dropout = 0.4
linear_units = [(bi+1) * lstm_units[-1], 128, 2]

### reguralization
l2_regulizer = 1e-1


model = nn.Sequential(
          Res_net(dropout=res_dropout, res_ch=res_ch, adaptive_seq_len=pooling),
          LSTM(num_layers=1,
               in_size=lstm_units[0],
               hidden_size=lstm_units[1],
               batch_size=batch_size,
               dropout=lstm_dropout,
               bidirectional=bi,
               return_seq=False,
               gpu=True,
               continue_seq=False),
          nn.Linear(linear_units[0], linear_units[1]),
          nn.BatchNorm1d(linear_units[1]),
          nn.Dropout(dense_dropout),
          nn.Linear(linear_units[1], linear_units[2]),
          nn.Softmax()
          )

model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             weight_decay=l2_regulizer,
                             lr = lr)
optimizer = Eve(model.parameters(),
                weight_decay=l2_regulizer,
                lr = lr)

##for tensorboard
# r = model(Variable(torch.ones(train_X.shape).cuda()))
writer = SummaryWriter()
# writer.add_graph(model, r)

### train_loop
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer)
trainer.run(epochs=epochs)
#
# trainer = ModuleTrainer(model)
# callbacks = [EarlyStopping(patience=10),
#              ReduceLROnPlateau(factor=0.5, patience=5)]
# metrics = [CategoricalAccuracy(top_k=2)]
# trainer.compile(loss='nll_loss',
#                 optimizer='adadelta',
#                 metrics=metrics)
# #summary = trainer.summary([1,28,28])
# #print(summary)
# trainer.fit(torch.from_numpy(train_X).float().cuda(), torch.from_numpy(train_y).long().cuda(),
#             val_data=(torch.from_numpy(test_X).float().cuda(), torch.from_numpy(test_y).long().cuda()),
#             num_epoch=128,
#             batch_size=32,
#             verbose=1)
#
# from skorch.net import NeuralNetClassifier
#
# net = NeuralNetClassifier(model, max_epochs=100, lr=1e-3)
# net.fit(X_crop, y_crop, batch_size=4096, use_cuda=True)
