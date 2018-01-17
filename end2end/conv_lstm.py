import numpy as np
import os, sys
os.chdir('/home/seigyo/Documents/pytorch/end2end')
sys.path.append(os.pardir)
from load_foot import Load_data, make_data
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from mymodule.layers import LSTM, T_CNN
from tensorboardX import SummaryWriter
from mymodule.trainer import Trainer
from mymodule.layers import Bayes_classifier, Residual_block, Res_net
from mymodule.utils import data_loader, evaluator, num_of_correct
from torch import nn
from sklearn.utils import shuffle
from torch_sample.torchsample.modules.module_trainer import ModuleTrainer

seq_len = 128

### preprocess 'ica' or 'pca'
preprocess= None
whiten = True

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

# train_x, train_t, test_x, test_t = data.corr_data(seq_len)
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

train_ = train_.reshape(train_.shape[0],
                        1,
                        train_.shape[1],
                        train_.shape[2])
test_ = test_.reshape(test_.shape[0],
                        1,
                        test_.shape[1],
                        test_.shape[2])

train_ = train_[:,:,:,0:5]
test_ = test_[:,:,:,0:5]

'''
validationとtrainingの分割
'''

train_x, train_tt = shuffle(train_, train_t)
cut = int(np.ceil(train_.shape[0]*(5/7)))
print(cut)
train_ = train_x[:cut]
val_ = train_x[cut:]
train_t = train_tt[:cut]
val_t = train_tt[cut:]

print(val_.shape)
print(train_.shape)
print(test_.shape)


f_dim = train_.shape[3]
seq_len = train_.shape[2]
'''
モデル
'''
epochs = 60
batch_size = 32
lr = 1e-6
train_loader = data_loader(train_, train_t, batch_size=batch_size,
                           shuffle=True, gpu=False)
test_loader = data_loader(test_, test_t, batch_size=batch_size)
val_loader = data_loader(val_, val_t, batch_size=batch_size)

class Conv_lstm(nn.Module):
  def __init__(self):
    super(Conv_lstm, self).__init__()
    self.conv_time = nn.Conv2d(1, 40, (25, 1))
    self.conv_spat = nn.Conv2d(40, 40, (1, 5), bias=False)
    self.batchnorm = nn.BatchNorm2d(40)
    self.pool = nn.AvgPool2d(kernel_size=(25, 1), stride=(5, 1))
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
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()

### train_loop
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer, gpu=True)
trainer.run(epochs=epochs)

model.eval()
test_y = nn.functional.softmax(model(Variable(torch.from_numpy(test_)).float().cuda()))
test_y.size()
corrects = num_of_correct(test_y, Variable(torch.from_numpy(test_t).long()).cuda())
accuracy = corrects / test_y.size(0)
print('accuracy:{}'.format(accuracy))

# loss, acc, pre_array = evaluator(model, criterion, test_loader, gpu=True)
# print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
#       test_.shape[0] , acc))

dt = 1/128
N = test_.shape[0]
t = np.linspace(1, N, N) * dt - dt

plt.plot(t, test_y.data.cpu().numpy()[:,0])
plt.plot(t, test_t)
plt.grid()
plt.xlabel('time')
plt.ylabel('predict and label')
plt.title('test infer')
plt.show()
