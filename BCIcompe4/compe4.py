import os, sys
os.chdir('/home/hikaru/Documents/pytorch/BCIcompe4')
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.optim as optim #最適化関数
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.pardir)
from tensorboardX import SummaryWriter
from mymodule.trainer import Trainer
from mymodule.layers import BayesLSTM, Bayes_classifier
from mymodule.utils import data_loader, evaluator
from sklearn.utils import shuffle
from torchqrnn import QRNN



def load_eeg(subject='a', split=0.8):
  ## calib_data
  ## 8秒置きに4秒間 MI
  data = loadmat('BCICIV_calib_ds1'+subject+'.mat')
  eeg = data['cnt'].astype(np.float32) * 0.1
  pos = data['mrk']['pos'][0][0][0]
  label = data['mrk']['y'][0][0][0].astype(np.float32)
  fs = data['nfo'][0][0][0][0][0].astype(np.float32)
  clab = []
  for i in range(59):
    clab.append(data['nfo']['clab'][0][0][0][i][0])
  class0 = data['nfo']['classes'][0][0][0][0][0]
  class1 = data['nfo']['classes'][0][0][0][1][0]
  xpos = data['nfo']['xpos'][0][0]
  ypos = data['nfo']['ypos'][0][0]

  print(class0, class1)
  target = np.zeros(shape=(eeg.shape[0],)).astype(np.float32)
  for i in range(len(pos)):
    t = label[i]
    idx = pos[i]
    target[idx:idx+400] = t

  target = target + 1

  X, T = shuffle(eeg, target)
  cut = int(np.ceil(eeg.shape[0]*(0.8)))
  eeg = X[:cut]
  target = T[:cut]
  valeeg = X[cut:]
  valtarget = T[cut:]
  return eeg, target, valeeg, valtarget

eeg1, target1, valeeg1, valtarget1 = load_eeg(number='a', split=0.8)
eeg2, target2, valeeg2, valtarget2 = load_eeg(number='b', split=0.8)
eeg3, target3, valeeg3, valtarget3 = load_eeg(number='c', split=0.8)
eeg4, target4, valeeg4, valtarget4 = load_eeg(number='d', split=0.8)
eeg5, target5, valeeg5, valtarget5 = load_eeg(number='e', split=0.8)

eeg = np.r_[eeg2, eeg3, eeg4, eeg5]
valeeg = np.r_[valeeg2, valeeg3, valeeg4, valeeg5]
target = np.r_[target2, target3, target4, target5]
valtarget = np.r_[valtarget2, valtarget3, valtarget4, valtarget5]


train_loader = data_loader(eeg, target, batch_size=2048,
                           shuffle=True, gpu=False)
val_loader = data_loader(valeeg, valtarget, batch_size=2048)



class Res_dense(nn.Module):
  def __init__(self, in_units, hidden_units, out_units, dropout):
    super(Res_dense, self).__init__()
    self.bn1 = nn.BatchNorm1d(in_units)
    self.l1 = nn.Linear(in_units, hidden_units)
    self.bn2 = nn.BatchNorm1d(hidden_units)
    self.l2 = nn.Linear(hidden_units, hidden_units)
    self.bn3 = nn.BatchNorm1d(hidden_units)
    self.l3 = nn.Linear(hidden_units, out_units)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    h1 = F.relu(self.dropout(self.l1(self.bn1(x))))
    h2 = F.relu(self.dropout(self.l2(self.bn2(h1))))
    y = F.relu(self.dropout(self.l3(self.bn3(h2))))
    return y + h1

model = nn.Sequential(
          Res_dense(59, 128, 128, 0.3),
          Res_dense(128, 256, 256, 0.2),
          Res_dense(256, 512, 512, 0.1),
          nn.BatchNorm1d(512),
          nn.Linear(512, 3)
          )


model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=None)
trainer.run(epochs=500)

# bayes_predictor = Bayes_classifier(predictor=model, num_sample=100)

data = loadmat('BCICIV_calib_ds1f.mat')
testeeg = data['cnt'].astype(np.float32) * 0.1
pos = data['mrk']['pos'][0][0][0]
label = data['mrk']['y'][0][0][0].astype(np.float32)
fs = data['nfo'][0][0][0][0][0].astype(np.float32)
clab = []
for i in range(59):
  clab.append(data['nfo']['clab'][0][0][0][i][0])
class0 = data['nfo']['classes'][0][0][0][0][0]
class1 = data['nfo']['classes'][0][0][0][1][0]
xpos = data['nfo']['xpos'][0][0]
ypos = data['nfo']['ypos'][0][0]

testtarget = np.zeros(shape=(testeeg.shape[0])).astype(np.float32)
for i in range(len(pos)):
  t = label[i]
  idx = pos[i]
  testtarget[idx:idx+400] = t

testtarget += 1

test_loader = data_loader(testeeg, testtarget, batch_size=2048)

loss, acc, pre_array = evaluator(model, criterion, test_loader)
print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
      valeeg.shape[0] , acc))

%matplotlib inline
plt.plot(np.argmax(pre_array, axis=1))
plt.plot(testtarget)
