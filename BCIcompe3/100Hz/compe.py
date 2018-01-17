import os, sys
os.chdir('/home/hikaru/Documents/pytorch/BCIcompe/100Hz')
import numpy as np
from data_loader import bpf, hpf, lpf, Get_data
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

data = Get_data()
plt.plot(data.eeg_select[100:300,:])
plt.show()

lpf1 = lpf(data.eeg_select, 36, 40, 100)
plt.plot(lpf1[100:300,:])
plt.show()

hpf1 = hpf(data.eeg_select, 2, 1, 100)
plt.plot(hpf1[:300,:])
plt.show()

bpf1 = bpf(data.eeg_select, [10.,15], [5, 20], 100)
plt.plot(bpf1[100:300,:])
plt.show()

train, label = data.get_time_array()

train.shape
