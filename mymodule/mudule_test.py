import os, sys
os.chdir('/home/seigyo/Documents/pytorch')
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mymodule.layers import ConvLSTM, ConvLSTMCell, SeqConvLSTM, NlayersSeqConvLSTM
%matplotlib inline

'''
convLSTMはseq_lenを考慮していない実装になっている。
layersを重ねることは可能だが、一時刻を計算するのみになっている。
２次元配列データの時系列（動画）を入れる場合には、ConvLSTMCellを使う。
NlayersSeqConvLSTMを実装
'''

cell = ConvLSTMCell(input_channels=1,
                    hidden_channels=16,
                    kernel_size=3,
                    bias=False)

h, c = cell.init_hidden(5, 16, (32, 32))
cell.cuda()

X = Variable(th.randn(5, 1, 32, 32)).cuda()
Y, c = cell(X, h, c)
Y.size()


model = NlayersSeqConvLSTM(input_channels=1,
                          hidden_channels=[64, 128, 256],
                          kernel_sizes=[3, 3, 3],
                          bias=False)
model.cuda()

X = Variable(th.randn(5, 16, 1, 32, 32)).cuda()
Y, all_Y = model(X)
all_Y[-1].size()
Y.size()
all_Y[-1].cpu().data.numpy() == Y.cpu().data.numpy()

model = SeqConvLSTM(input_channels=1,
                    hidden_channels=64,
                    kernel_size=3,
                    bias=False)
model.cuda()
model2 = SeqConvLSTM(input_channels=64,
                    hidden_channels=128,
                    kernel_size=3,
                    bias=False)
model2.cuda()
X = Variable(th.randn(5, 16, 1, 32, 32)).cuda()
Y = model2(model(X))
Y.size()
