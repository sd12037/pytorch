import numpy as np
from scipy import *
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FastICA
## data import

class Load_data():

    def __init__(self, train_mat, test_mat, train_label_mat, test_label_mat):


        def load_data(matfile):
            data = loadmat(matfile)
            return data['eeg_foot_5ch'].astype(np.float32)

        self.train = load_data(train_mat)
        self.test = load_data(test_mat)

        def load_label(matfile):
            data = loadmat(matfile)
            label = data['C'].astype(np.float32)
            for i in range(len(label)):
                if label[i] == 2:
                    label[i] = 1
                if label[i] == 3:
                    label[i] = 0
            return np.c_[label, 1 - label]

        self.train_label = load_label(train_label_mat)
        self.test_label = load_label(test_label_mat)

    def pca(self, whiten = True):
        pca = PCA(n_components = 5, whiten = whiten)
        pca.fit(self.train)
        self.train = pca.transform(self.train)
        self.test = pca.transform(self.test)

    def ica(self, whiten = True):
        ica = FastICA(n_components = 5, whiten = whiten)
        ica.fit(self.train)
        self.train = ica.transform(self.train)
        self.test = ica.transform(self.test)


    def get_data2d(self, seq_len):
        '''
        seq_len x fearture で１つのデータとする
        '''

        def trainspose2D(train_data, label_data):
            num_batch = len(train_data) - seq_len + 1
            x = np.zeros((num_batch, seq_len, 5))
            for start in range(len(train_data) - seq_len + 1):
                x[start, :, :] = train_data[start: start + seq_len]
            label_data = label_data[:num_batch]
            return x, label_data

        train, train_label = trainspose2D(self.train, self.train_label)
        test, test_label = trainspose2D(self.test, self.test_label)
        return train, train_label, test, test_label

    def get_data1d(self):
        return self.train, self.test, self.train_label, self.test_label

    def split(self, rate = 0.7):
        cut_idx = int(np.ceil(rate * len(self.train)))
        train, label = shuffle(self.train, self.label)
        self.train = train[:cut_idx]
        self.train_label = label[:cut_idx]
        self.val = train[cut_idx:]
        self.val_label = label[cut_idx:]
        return self.train, self.train_label, self.val, self.val_label

    def get_data_var_seq(self, seq_len):
        '''
        elec x elec x 1 で １つのデータ, 電極間の共分散行列の構成要素１つ１つをデータとする。
        '''

        def trainspose4D(train_data, label_data):
            num_batch = len(train_data) - seq_len + 1
            x = np.zeros((num_batch, seq_len, 5, 5, 1))
            for start in range(len(train_data) - seq_len + 1):
                x[start, :, :, :, :] = train_data[start: start + seq_len]
            label_data = label_data[:num_batch]
            return x, label_data

        train = np.zeros((14336, 5, 5, 1))
        test = np.zeros((14336, 5, 5, 1))

        for i in range(train.shape[0]):
            train[i, :, :, 0] = np.dot(self.train[i].T, self.train[i])
            test[i, :, :, 0] = np.dot(self.test[i].T, self.test[i])

        train_var, train_label = trainspose4D(train, self.train_label)
        test_var, test_label = trainspose4D(test, self.test_label)

        return train_var, train_label, test_var, test_label


    def corr_data(self, seq_len):
        '''
        自己相関系列へ変換
        '''

        def normalization(x):
            mu = np.mean(x)
            sigma = np.sqrt(np.var(x))
            xx = x/sigma - mu
            return xx

        train, train_label, test, test_label = self.get_data2d(seq_len)

        new_train = np.zeros(train.shape)
        new_test = np.zeros(test.shape)

        for i in range(new_train.shape[0]):
            for j in range(new_train.shape[2]):
              b_train = normalization(train[i,:,j])
              b_test = normalization(test[i,:,j])
              corr_train = np.correlate(b_train, b_train, 'same')
              corr_test = np.correlate(b_test, b_test, 'same')
              new_train[i,:,j] = corr_train
              new_test[i,:,j] = corr_test

        return new_train, train_label, new_test, test_label



def make_data(train_mat="train_foot2.mat",
              test_mat="test_foot2.mat",
              train_label_mat="label_foot.mat",
              test_label_mat="label_foot.mat",
              seq_len=64):
    data = Load_data(train_mat=train_mat,
                     test_mat=test_mat,
                     train_label_mat=train_label_mat,
                     test_label_mat=test_label_mat)
    train, train_label, test, test_label = data.get_data2d(seq_len)
    return train, train_label, test, test_label
