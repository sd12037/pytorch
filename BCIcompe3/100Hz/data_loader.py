import numpy as np
import shutil
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

def bpf(x, pass_freq=[3., 35.], cut_freq=[0.5, 40.], sample_rate=100):
  wp = list(np.array(pass_freq)/(sample_rate/2))
  ws = list(np.array(cut_freq)/(sample_rate/2))
  N, Wn = signal.buttord(wp, ws, 3, 40, False)
  b, a = signal.butter(N, Wn, btype='bandpass', analog=False, output='ba')
  y = signal.lfilter(b, a, x)
  return y

def lpf(x, pass_freq=35., cut_freq=40., sample_rate=100):
  wp = pass_freq/(sample_rate/2)
  ws = cut_freq/(sample_rate/2)
  N, Wn = signal.buttord(wp, ws, 3, 40, False)
  b, a = signal.butter(N, Wn, btype='low', analog=False, output='ba')
  y = signal.lfilter(b, a, x)
  return y

def hpf(x, pass_freq=35., cut_freq=30., sample_rate=100):
  wp = pass_freq/(sample_rate/2)
  ws = cut_freq/(sample_rate/2)
  N, Wn = signal.buttord(wp, ws, 3, 40, False)
  b, a = signal.butter(N, Wn, btype='high', analog=False, output='ba')
  y = signal.lfilter(b, a, x)
  return y


class Get_data():
  def __init__(self):
    data = loadmat('data_set_IVb_al_train.mat')

    eeg = data['cnt'].astype(np.float32) * 0.1
    self.pos = data['mrk']['pos'][0][0][0]
    self.label = data['mrk']['y'][0][0][0].astype(np.float32)

    for i in range(118):
        electrode = data['nfo']['clab'][0][0][0][i][0]
        if electrode == 'Cz':
            Cz_idx = i
        elif electrode == 'C1':
            C1_idx = i
        elif electrode == 'C2':
            C2_idx = i
        elif electrode == 'C3':
            C3_idx = i
        elif electrode == 'C4':
            C4_idx = i
        elif electrode == 'C5':
            C5_idx = i
        elif electrode == 'C6':
            C6_idx = i
        elif electrode == 'FCz':
            FCz_idx = i
        elif electrode == 'CPz':
            CPz_idx = i
        elif electrode == 'FC3':
            FC3_idx = i
        elif electrode == 'CP3':
            CP3_idx = i
        elif electrode == 'FC4':
            FC4_idx = i
        elif electrode == 'CP4':
            CP4_idx = i
    select_idx = [Cz_idx, C1_idx, C2_idx, FCz_idx, CPz_idx,
                  C3_idx, FC3_idx, CP3_idx, C5_idx,
                  C4_idx, FC4_idx, CP4_idx, C6_idx]
    self.select_idx = select_idx
    self.eeg_select = eeg[:,select_idx]

  def get_time_array(self, seq_len=100):
    pos_lis = list(self.pos)
    train = np.zeros((210, seq_len, self.eeg_select.shape[1]))
    for idx in range(len(pos_lis)):
        start = pos_lis[idx]
        end = pos_lis[idx] + seq_len
        train[idx, :, :] = self.eeg_select[start: end]

    label = (self.label + 0.8).astype(np.int32).astype(np.float32)


    return train, label

  def small_laplacian_filter(self):
    small_laplacian_filter = np.zeros((self.eeg_select.shape[0], 3))
    small_laplacian_filter[:,0] = self.eeg_select[:,0] \
                                  - 0.25 * (self.eeg_select[:,1] +
                                            self.eeg_select[:,2] +
                                            self.eeg_select[:,3] +
                                            self.eeg_select[:,4])
    small_laplacian_filter[:,1] = self.eeg_select[:,5] \
                                  - 0.25 * (self.eeg_select[:,1] +
                                            self.eeg_select[:,6] +
                                            self.eeg_select[:,7] +
                                            self.eeg_select[:,8])
    small_laplacian_filter[:,2] = self.eeg_select[:,9] \
                                  - 0.25 * (self.eeg_select[:,2] +
                                            self.eeg_select[:,10] +
                                            self.eeg_select[:,11] +
                                            self.eeg_select[:,12])

    self.eeg_select = small_laplacian_filter

  def bandpass(self, pass_freq=[3., 35.],
               cut_freq=[0.5, 40.], sample_rate=100):

    self.eeg_select = bpf(self.eeg_select, pass_freq=[3., 35.],
                          cut_freq=[0.5, 40.], sample_rate=100)

if __name__ == '__main__':

  # train, label = Get_data().get_time_array()
  # print(train.shape)
  # print(label.shape)
  data = Get_data()
  data.small_laplacian_filter()
  small, label = data.get_time_array(100)
  print(small.shape)
