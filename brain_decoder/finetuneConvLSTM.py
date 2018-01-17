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
from mymodule.utils import num_of_correct, evaluator
from mymodule.layers import LSTM, Residual_block, Res_net, Wavelet_cnn, NlayersSeqConvLSTM
from sklearn.utils import shuffle
from load_data import get_data, get_data_multi, get_crops, get_crops_multi,get_data_one_class
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

def time_normalization(x, time_dim):
    return (x-x.mean(dim=time_dim, keepdim=True))/(torch.sqrt(x.var(dim=time_dim, keepdim=True))+1e-8)

def elec_map2d(X):
    batch_size = X.shape[0]
    X_spat = np.zeros((batch_size, X.shape[-1], 1, 5, 7))
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
                                hidden_channels=[64],
                                kernel_sizes=[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        h = self.conv_time(x)
        h = self.pad(h)
        h = F.elu(self.conv_spat(h))
        h = self.batchnorm(h)
        h = self.dropout(h)

        h = self.conv_time2(h)
        h = self.pad(h)
        h = F.elu(self.conv_spat2(h))
        h = self.batchnorm2(h)
        h = self.dropout2(h)

        h = self.conv_time3(h)
        h = self.pad(h)
        h = F.elu(self.conv_spat3(h))
        h = self.batchnorm3(h)
        h = self.pool_time3(h)
        h = self.dropout3(h)

        h = h.transpose(1, 2) # (batch, ch, seq, h, w) -> (batch, seq, ch, h, w)
        h, _ = self.convlstm(h)
        h = h[:,-1,:,:,:] # last seq
        h = self.avgpool(h).squeeze()
        h = self.batchnorm4(h)
        h = self.dropout4(h)
        return self.linear(h)


def make_class(subject_id=[1,10], crop=False, problem='hf', all_subject=False):
    bpfilter = [0.5, 45]
    if problem == "hf":
        if crop:
            pass
        else:
            X, y = get_data_multi(sub_id_range=subject_id, event_code=[6,10,14], t=[0, 4.0], filter=bpfilter)
    if problem == "lr":
        if crop:
            X, y, _, _ = get_crops_multi(sub_id_range=subject_id, event_code=[4,8,12], t=[0, 4.0], filter=bpfilter,
                    time_window=2.0, time_step=1.0)
        else:
            X, y = get_data_multi(sub_id_range=subject_id, event_code=[4,8,12], t=[0, 4.0], filter=bpfilter)
    return X, y


class Trainer(object):
  def __init__(self, model=None, criterion=None, optimizer=None,
               train_loader=None, val_loader=None, val_num=1,
               gpu=True):

    self.model = model
    self.gpu = gpu
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.epoch = 0
    self.val_num = val_num
    self.train_best_acc = None
    self.train_best_loss = None
    self.val_best_acc = None
    self.val_best_loss = None
    self.acc_list = []
    self.val_acc_list = []
  def train(self):


    tr_runnning_loss = 0.
    tr_runnning_correct = 0.
    num_data = 0
    self.epoch += 1
    n_iter = 1
    for x, y in self.train_loader:

      if self.gpu:
        tr_batch_x = Variable(x.cuda())
        tr_batch_y = Variable(y.cuda())
      else:
        tr_batch_x = Variable(x)
        tr_batch_y = Variable(y)
      self.optimizer.zero_grad()
      outputs = self.model(tr_batch_x)
      num_data += outputs.data.shape[0]
      loss = self.criterion(outputs, tr_batch_y)
      corrects = num_of_correct(outputs, tr_batch_y)
      loss.backward()
      self.optimizer.step()
      n_iter += 1
      tr_runnning_loss += loss.data[0]
      tr_runnning_correct += corrects

    tr_runnning_loss /= n_iter
    training_acc = tr_runnning_correct/num_data
    self.acc_list.append(training_acc)
    if self.epoch == 1:
      self.train_best_acc = training_acc
      self.train_best_loss = tr_runnning_loss

    if self.train_best_acc < training_acc:
      self.train_best_acc = training_acc

    if self.train_best_loss > tr_runnning_loss:
      self.train_best_loss = tr_runnning_loss

    # self.writer.add_scalar('training loss', tr_runnning_loss, self.epoch)
    # self.writer.add_scalar('training accuracy', training_acc, self.epoch)

    print('epoch:{}, tr_loss:{:0.4f}, tr_acc:{:0.4f},   '.format(
           self.epoch, tr_runnning_loss, training_acc), end='')

    # if self.epoch % self.val_num == 0:
      # self.val_loss, val_accuracy = evaluator(self.model,
      #                       self.criterion, self.val_loader)

    val_runnning_loss = 0.
    val_runnning_correct = 0.
    val_num_data = 0
    self.model.eval()
    n_iter = 1
    for x, y in self.val_loader:
      if self.gpu:
        batch_x = Variable(x.cuda(), volatile=True)
        batch_y = Variable(y.cuda())
      else:
        batch_x = Variable(x, volatile=True)
        batch_y = Variable(y)
      outputs = self.model(batch_x)
      val_num_data += outputs.data.shape[0]
      val_loss = self.criterion(outputs, batch_y)
      corrects = num_of_correct(outputs, batch_y)

      n_iter += 1
      val_runnning_loss += val_loss.data[0]
      val_runnning_correct += corrects

    val_runnning_loss /= n_iter
    val_acc = val_runnning_correct/val_num_data
    self.val_acc_list.append(val_acc)

    if self.epoch == 1:
      self.val_best_acc = val_acc
      self.val_best_loss = val_runnning_loss

    if self.val_best_acc < val_acc:
      self.val_best_acc = val_acc
      self.model_saver()

    if self.val_best_loss > val_runnning_loss:
      self.val_best_loss = val_runnning_loss
    # self.writer.add_scalar('validation loss', val_runnning_loss, self.epoch)
    # self.writer.add_scalar('validation accuracy', val_acc, self.epoch)


    print('val_loss:{:0.4f}, val_acc:{:0.4f}'.format(
            val_runnning_loss, val_acc))

    self.model.train()

  def run(self, epochs=100):
    print('----------start training----------')
    for _ in range(epochs):
      self.train()
    print('----------finish training---------')
    print('training_best_acc:{}, val_best_acc:{}'.format(
                self.train_best_acc, self.val_best_acc))
    np.save('train_acc', np.array(self.acc_list))
    np.save('val_acc', np.array(self.val_acc_list))
      # if self.early_stopping.validate(self.val_loss):
      #
      #   break

  def model_saver(self):
    torch.save(model.state_dict(), 'weight_convLSTM.pth')



 
X, y = make_class(subject_id=[1,50], crop=False, problem='hf', all_subject=False)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X, y = shuffle(X, y)
X = elec_map2d(X)
print(X.shape)
X = X.transpose(0,2,1,3,4)
print(X.shape)
# Split dataset
# Train : Test = 8 : 2
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2)

train_loader = data_loader(X_train, y_train, batch_size=128,
                           shuffle=True, gpu=False)
val_loader = data_loader(X_test, y_test, batch_size=128)
model = Conv3d_convLSTM().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, criterion, optimizer,
            train_loader, val_loader,
            val_num=1, gpu=True, )
trainer.run(epochs=400)