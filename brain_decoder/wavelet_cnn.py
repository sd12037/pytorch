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
from mymodule.layers import Wavelet_cnn
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from load_data import get_data, get_data_multi
from skorch.net import NeuralNetClassifier
from sklearn.model_selection import train_test_split

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
X, y = get_data(id=1, event_code=[5,9,13], filter=None, t=[-0.5, 4])
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).transpose(0,1,3,2)
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.2)

# train_X, train_y = X[:30], y[:30]
# test_X, test_y = X[30:], y[30:]

_, _, seq_len, f_dim = X_train.shape

epochs = 200
batch_size = 16

class Mydnn(nn.Module):
  def __init__(self, time_window=2, stride=2, n_elec=64):
    super(Mydnn, self).__init__()
    self.wavelet = Wavelet_cnn(time_window, stride)
    self.conv1 = nn.Conv2d(n_elec, 128, (25, 2))
    self.conv2 = nn.Conv2d(128, 256, (25, 2))
    self.conv3 = nn.Conv2d(256, 512, (25, 2))
    self.pool = nn.AvgPool2d((2, 1))
    self.conv_class = nn.Conv2d(512, 2, (1, 1))
    self.pool_class = nn.AdaptiveAvgPool2d(output_size=(1, 1))

  def forward(self, x):
    scalegram, _ = self.wavelet(x)
    scalegram = scalegram ** 2 # power
    scalegram = scalegram.transpose(1, 3) # (batch, elec, seq, level)
    h = self.pool(F.relu(self.conv1(scalegram)))
    h = self.pool(F.relu(self.conv2(h)))
    h = self.pool(F.relu(self.conv3(h)))
    h = F.softmax(self.pool_class(self.conv_class(h)).squeeze())
    return h

model = NeuralNetClassifier(module=Mydnn,
                            max_epochs=100,
                            lr=1e-4)

model.fit(X_train, y_train)
