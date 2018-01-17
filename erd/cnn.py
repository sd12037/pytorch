import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連
import torchvision #画像関連
import numpy as np
from get_data import get_data
from sklearn.utils import shuffle
import torch.cuda
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.pardir)
from mymodule.trainer import Trainer
from mymodule.utils import data_loader, evaluator
from tensorboardX import SummaryWriter

writer = SummaryWriter()

batch_size = 1024
epochs = 100
'''
データの生成
'''
train, test, label = get_data(idx=1)
train_loader = data_loader(train, label, batch_size=batch_size,
                           shuffle=True, gpu=False)
test_loader = data_loader(test, label, batch_size=batch_size,
                           shuffle=False, gpu=False)

class MLP(nn.Module):
  def __init__(self, training=True):
    super(MLP, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, padding=1,
                           kernel_size=(2, 2), bias=False)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, padding=1,
                           kernel_size=(2, 2), bias=False)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=1, padding=1,
                           kernel_size=(2, 2), bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(1)
    self.do1 = nn.Dropout2d(0.3)
    self.do2 = nn.Dropout2d(0.3)
    self.do3 = nn.Dropout2d(0.3)
    self.Ada_max_Pool2d = nn.AdaptiveMaxPool2d((5, 5))

    self.lstm = nn.LSTM(input_size=5, hidden_size=10, num_layers=1,
                        dropout=.1, bidirectional=True, batch_first=True)

    self.l1 = nn.Linear(100, 512)
    self.l2 = nn.Linear(512, 128)
    self.l3 = nn.Linear(128, 2)
    self.bnl1 = nn.BatchNorm1d(512)
    self.bnl2 = nn.BatchNorm1d(128)
    self.dol1 = nn.Dropout(0.3)
    self.dol2 = nn.Dropout(0.3)


  def forward(self, x):
    h = self.conv1(x)
    # h = self.bn1(h)
    h = self.do1(h)
    h = nn.functional.max_pool2d(h, kernel_size=(2, 2))
    h = self.conv2(h)
    # h = self.bn2(h)
    h = self.do2(h)
    h = nn.functional.max_pool2d(h, kernel_size=(2, 2))
    h = self.conv3(h)
    # h = self.bn3(h)
    h = self.do3(h)
    h = self.Ada_max_Pool2d(h)
    h = torch.squeeze(h, dim=1)

    self.h0 = Variable(torch.zeros(2, h.size(0), 10).cuda())
    self.c0 = Variable(torch.zeros(2, h.size(0), 10).cuda())
    h, _ = self.lstm(h, (self.h0, self.c0))

    h = h.contiguous()

    num_batch = h.size(0)
    h = h.view(num_batch, 100)
    h = self.l1(h)
    h = nn.functional.relu(h)
    h = self.dol1(h)
    h = self.l2(h)
    h = self.bnl2(h)
    h = nn.functional.relu(h)
    h = self.dol2(h)
    h = self.l3(h)
    return h


def num_of_correct(model, x, t):
    _, predicted = torch.max(model(x), 1)
    corrects = torch.sum(predicted.data == t.data)
    return corrects

model = MLP()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.00001)

out = model(torch.autograd.Variable(
      torch.cuda.FloatTensor(batch_size, train.shape[1],
                             train.shape[2], train.shape[3]),
                             requires_grad=True))
writer.add_graph(model, out)


trainer = Trainer(model, criterion, optimizer,
                  train_loader, test_loader,
                  val_num=5, early_stopping=None,
                  writer=writer)
trainer.run(epochs=epochs)



'''モデル評価'''


loss, acc, pre_array = evaluator(model, criterion, test_loader)
print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
      test.shape[0] , acc))

dt = 1/128
N = test.shape[0]
t = np.linspace(1, N, N) * dt - dt
plt.plot(t, label)
plt.plot(t, pre_array[:,1])
plt.grid()
plt.xlabel('time')
plt.ylabel('predict and label')
plt.title('test infer')
plt.show()
