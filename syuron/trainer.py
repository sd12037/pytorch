import torch
import numpy as np
from torch.autograd import Variable
from mymodule.utils import num_of_correct, evaluator

class Trainer(object):
  def __init__(self, model=None, criterion=None, optimizer=None,
               train_loader=None, val_loader=None,
               gpu=True, name=None):

    if name == None:
        self.name = input("please give a name to this model:")
    else:
        self.name = name
    self.model = model
    self.gpu = gpu
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.epoch = 0
    self.train_best_acc = None
    self.train_best_loss = None
    self.val_best_acc = None
    self.val_best_loss = None
    self.acc_list = []
    self.val_acc_list = []
    self.loss_list = []
    self.val_loss_list = []
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
    self.loss_list.append(tr_runnning_loss)

    if self.epoch == 1:
      self.train_best_acc = training_acc
      self.train_best_loss = tr_runnning_loss

    if self.train_best_acc < training_acc:
      self.train_best_acc = training_acc

    if self.train_best_loss > tr_runnning_loss:
      self.train_best_loss = tr_runnning_loss

    print('epoch:{}, tr_loss:{:0.4f}, tr_acc:{:0.4f},   '.format(
           self.epoch, tr_runnning_loss, training_acc), end='')


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
    self.val_loss_list.append(val_runnning_loss)

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
    np.save(self.name + 'train_acc', np.array(self.acc_list))
    np.save(self.name + 'val_acc', np.array(self.val_acc_list))
    np.save(self.name + 'train_loss', np.array(self.loss_list))
    np.save(self.name + 'val_loss', np.array(self.val_loss_list))
      # if self.early_stopping.validate(self.val_loss):
      #
      #   break
  def model_saver(self):
    acc = round(self.val_best_acc, 4)
    torch.save(self.model.state_dict(),
               self.name + str(self.epoch) + '_'+ str(acc) +  '.pth')