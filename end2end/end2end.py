import numpy as np
import cupy as cp
import os, sys
sys.path.append(os.pardir)
from load_foot import Load_data, make_data
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from mymodule.layers import LSTM, T_CNN
from tensorboardX import SummaryWriter
from mymodule.trainer import Trainer
from mymodule.layers import Bayes_classifier, Residual_block, Res_net
from mymodule.utils import data_loader, evaluator
from torch import nn
from sklearn.utils import shuffle
import tensorflow as tf

epochs = 100
seq_len = 128
batch_size = 32

### preprocess 'ica' or 'pca'
preprocess= None
whiten = True

### resnet
res_ch = [10, 32]
pooling = [int(seq_len/2), int(seq_len/4)]
res_dropout = 0.5

### lstm
lstm_units = [res_ch[-1], 32]
lstm_dropout = 0.5
bi = False

### linear
linear_units = [(bi+1) * lstm_units[-1], 128, 2]
dense_dropout = 0.5

### reguralization
l2_regulizer = 1e-1
bayes = True
sample = 100

if bayes==False:
  sample = 1


'''
データ前処理
'''
os.chdir('data')
data = Load_data(train_mat='train_foot.mat',
                 test_mat='test_foot.mat',
                 train_label_mat='label_foot.mat',
                 test_label_mat='label_foot.mat')
os.chdir('..')

if preprocess=='pca':
  data.pca(whiten = whiten)
if preprocess=='ica':
  data.ica(whiten = whiten)
else:
  pass

train_x, train_t, test_x, test_t = data.corr_data(seq_len)
train_x, train_t, test_x, test_t = data.get_data2d(seq_len)

'''
入力データ作成
'''
train_t = train_t[:,1]
test_t = test_t[:,1]

train_diff = np.zeros(train_x.shape)
test_diff = np.zeros(test_x.shape)
train_diff[:,1:seq_len,:] = np.diff(train_x, axis=1)
test_diff[:,1:seq_len,:] = np.diff(test_x, axis=1)

### (batch, seq, 10 = 5 + 5)
train_ = np.c_[train_x, train_diff]
test_ = np.c_[test_x, test_diff]

train_ = train_.transpose(0,2,1)
test_ = test_.transpose(0,2,1)

'''
validationとtrainingの分割
'''

train_x, train_tt = shuffle(train_, train_t)
cut = int(np.ceil(train_.shape[0]*(2/7)))
print(cut)
train_ = train_x[:cut]
val_ = train_x[cut:]
train_t = train_tt[:cut]
val_t = train_tt[cut:]

print(val_.shape)
print(train_.shape)

train_loader = data_loader(train_, train_t, batch_size=batch_size,
                           shuffle=True, gpu=False)
# test_loader = data_loader(test_, test_t, batch_size=batch_size)
val_loader = data_loader(val_, val_t, batch_size=batch_size)


f_dim = train_.shape[2]

'''
モデル
'''

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.res = Res_net(dropout=res_dropout, res_ch=res_ch, bayes=bayes)
    self.lstm = LSTM(num_layers=1,
                     in_size=lstm_units[0],
                     hidden_size=lstm_units[1],
                     batch_size=batch_size,
                     dropout=lstm_dropout,
                     bidirectional=bi,
                     return_seq=False,
                     gpu=True,
                     continue_seq=False,
                     bayes=bayes)
    self.L1 = nn.Linear(linear_units[0], linear_units[1])
    self.L2 = nn.Linear(linear_units[1], linear_units[2])
    self.bn1 = nn.BatchNorm1d(linear_units[1])
    self.do1 = nn.Dropout(dense_dropout)

  def forward(self, x):
    h = self.res(x)
    h = self.lstm(h)
    h = self.L1(h)
    h = self.bn1(h)
    if bayes:
      h = nn.functional.dropout(h, p=dense_dropout, training=True)
    else:
      h = self.do1(h)
    h = self.L2(h)
    return h


model = MLP()
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2_regulizer)

##for tensorboard
# r = model(Variable(torch.ones(train_.shape).cuda()))
writer = SummaryWriter()
# writer.add_graph(model, r)

### train_loop
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer)
trainer.run(epochs=epochs)

#
# bayes_predictor = Bayes_classifier(predictor=model, num_sample=sample)
#
# loss, acc, pre_array = evaluator(bayes_predictor, criterion, test_loader)
# print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
#       test_x.shape[0] , acc))
#
#
# dt = 1/128
# N = test_x.shape[0]
# t = np.linspace(1, N, N) * dt - dt
#
# plt.plot(t, test_t)
# plt.plot(t, pre_array[:,1])
# plt.grid()
# plt.xlabel('time')
# plt.ylabel('predict and label')
# plt.title('test infer')
# plt.show()
