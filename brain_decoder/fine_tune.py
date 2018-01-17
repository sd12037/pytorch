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

class FBCSP(nn.Module):
  def __init__(self):
    super(FBCSP, self).__init__()
    self.time_normalization = time_normalization
    self.conv_time = nn.Conv2d(1, 32, (75, 1))
    self.conv_spat = nn.Conv2d(32, 32, (1, 64), bias=False)
    self.batchnorm2 = nn.BatchNorm2d(32)
    self.pool = nn.AvgPool2d(kernel_size=(75, 1), stride=(25, 1))
    self.dropout = nn.Dropout2d(p=0.5)
    self.conv_class = nn.Conv2d(32, 2, (12, 1))
    self.score_ave = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    x = self.time_normalization(x, time_dim=2)
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
                    time_window=1.0, time_step=0.5)
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
    torch.save(model.state_dict(), 'lh_crops.pth')

X, y = make_class(subject_id=[1,50], crop=True, problem='lr', all_subject=False)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X, y = shuffle(X, y)
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).transpose(0,1,3,2)
# Split dataset
# Train : Test = 8 : 2
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2)

train_loader = data_loader(X_train, y_train, batch_size=256,
                           shuffle=True, gpu=False)
val_loader = data_loader(X_test, y_test, batch_size=256)
model = FBCSP().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, criterion, optimizer,
            train_loader, val_loader,
            val_num=1, gpu=True, )
trainer.run(epochs=500)
