import numpy as np
from data_loader import bpf, Get_data
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

data = Get_data()
# data.small_laplacian_filter()
data.bandpass(pass_freq=[3, 40], cut_freq=[1, 45], sample_rate=100)
train, label = data.get_time_array()


epochs = 1000
batch_size = 5
seq_len = train.shape[1]
f_dim = train.shape[2]
lstm1_dim = 500
