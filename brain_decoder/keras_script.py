import os, sys
os.chdir('/home/seigyo/Documents/pytorch/brain_decoder')
sys.path.append(os.pardir)
import numpy as np
from numpy.random import RandomState
from mne.io import concatenate_raws
from mymodule.utils import data_loader, evaluator
from mymodule.layers import LSTM, Residual_block, Res_net, Wavelet_cnn, NlayersSeqConvLSTM
from mymodule.trainer import Trainer
from sklearn.utils import shuffle
from load_data import get_data, get_data_multi, get_crops, get_crops_multi,get_data_one_class
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras import backend as K

def model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(75,1),input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(1,64)))
    model.add(AveragePooling2D(pool_size=(75, 1),stride=(25,1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(2), activation='softmax')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def make_class(idx=0, crop=False, problem='hf'):
    if problem == "hf":
        if crop:
            pass
        else:
            X, y = get_data(id=idx+1, event_code=[6,10,14], filter=[0.5, 45], t=[0., 4])
    if problem == "lr":
        if crop:
            pass
        else:
            X, y = get_data(id=idx+1, event_code=[4,8,12], filter=[0.5, 45], t=[0., 4])
    if problem == "lh":
        if crop:
            pass
        else:
            Xl, yl = get_data_one_class(id=idx+1, event_code=[4,8,12], filter=[0.5, 45], t=[0, 4], classid=2)
            Xh, yh = get_data_one_class(id=idx+1, event_code=[6,10,14], filter=[0.5, 45], t=[0, 4], classid=2)
            X = np.vstack((Xl,Xh))
            y = np.hstack((yl,yh+1))
    if problem == "rh":
        if crop:
            pass
        else:
            Xr, yr = get_data_one_class(id=idx+1, event_code=[4,8,12], filter=[0.5, 45], t=[0, 4], classid=3)
            Xh, yh = get_data_one_class(id=idx+1, event_code=[6,10,14], filter=[0.5, 45], t=[0, 4], classid=2)
            X = np.vstack((Xr,Xh))
            y = np.hstack((yr-1,yh+1))
    if problem == "lf":
        if crop:
            pass
        else:
            Xl, yl = get_data_one_class(id=idx+1, event_code=[4,8,12], filter=[0.5, 45], t=[0, 4], classid=2)
            Xf, yf = get_data_one_class(id=idx+1, event_code=[6,10,14], filter=[0.5, 45], t=[0, 4], classid=3)
            X = np.vstack((Xl,Xf))
            y = np.hstack((yl,yf))
    if problem == "rf":
        if crop:
            pass
        else:
            Xr, yr = get_data_one_class(id=idx+1, event_code=[4,8,12], filter=[0.5, 45], t=[0, 4], classid=3)
            Xf, yf = get_data_one_class(id=idx+1, event_code=[6,10,14], filter=[0.5, 45], t=[0, 4], classid=3)
            X = np.vstack((Xr,Xf))
            y = np.hstack((yr-1,yf))
    return X, y


def main():
    X, y = make_class(idx=0)
    model = model(input_shape=X.shape[2:])
    model.fit(X, y)
if __name__ == '__main__':
    main()