import numpy as np
from scipy.io import loadmat
import os, sys
sys.path.append(os.pardir)

'''
データの生成
'''
def get_data(idx=1):

  dataset = loadmat('train_data_CNN' + str(idx) + '.mat')
  train = dataset['train_data']
  train = np.array(train)
  train = train.astype(np.float32)
  train = train.reshape((len(train), 1, 28, 28,))

  dataset = loadmat('test_data_CNN'+ str(idx) + '.mat')
  test = dataset['train_data']
  test = np.array(test)
  test = test.astype(np.float32)
  test = test.reshape((len(train), 1, 28, 28))

  dataset = loadmat('label_data_CNN.mat')
  label = dataset['label_data']
  label = np.array(label).astype(np.int32)
  label = label.flatten()

  return train, test, label
