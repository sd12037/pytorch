import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, sys
sys.path.append(os.pardir)
from mymodule.utils import num_of_correct, evaluator
from tensorboardX import SummaryWriter
import torchnet as tnt



class Trainer(object):
  def __init__(self, model=None, criterion=None, optimizer=None,
               train_loader=None, val_loader=None, val_num=1,
               early_stopping=None, writer=None, gpu=True, log=False):

    if writer == None:
      self.writer = SummaryWriter()
    else:
      self.writer = writer
    self.model = model
    self.gpu = gpu
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.epoch = 0
    self.val_num = val_num
    # if early_stopping==None:
    #   self.early_stopping = EarlyStopping(patience=10, verbose=0)
    # else:
    #   self.early_stopping = EarlyStopping(patience=early_stopping, verbose=1)
    # self.val_loss = 100
    self.train_best_acc = None
    self.train_best_loss = None
    self.val_best_acc = None
    self.val_best_loss = None
    self.log = log

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

    if self.epoch == 1:
      self.val_best_acc = val_acc
      self.val_best_loss = val_runnning_loss

    if self.val_best_acc < val_acc:
      self.val_best_acc = val_acc

    if self.val_best_loss > val_runnning_loss:
      self.val_best_loss = val_runnning_loss
    # self.writer.add_scalar('validation loss', val_runnning_loss, self.epoch)
    # self.writer.add_scalar('validation accuracy', val_acc, self.epoch)

    if self.log:
      self.writer.add_scalars('loss',
                           {'training': tr_runnning_loss,
                            'validation': val_runnning_loss},
                            self.epoch)
      self.writer.add_scalars('accuracy',
                           {'training': training_acc,
                            'validation': val_acc},
                            self.epoch)


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
      # if self.early_stopping.validate(self.val_loss):
      #
      #   break

  def model_saver(self):
    pass



class EarlyStopping():
  def __init__(self, patience=0, verbose=0):
    self._step = 0
    self._loss = float('inf')
    self.patience = patience
    self.verbose = verbose

  def validate(self, loss):
    if self._loss < loss:
      self._step += 1
      if self._step > self.patience:
        if self.verbose:
          print('early stopping')
        return True
    else:
      self._step = 0
      self._loss = loss
    return False
