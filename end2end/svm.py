import numpy as np
from sklearn import svm
import os, sys
sys.path.append(os.pardir)
from load_foot import Load_data, make_data
import matplotlib.pyplot as plt

os.chdir('data')
data = Load_data(train_mat='train_foot.mat',
                 test_mat='test_foot.mat',
                 train_label_mat='label_foot.mat',
                 test_label_mat='label_foot.mat')
os.chdir('..')

# data.pca()
# data.ica()
train_x, train_t, test_x, test_t = data.corr_data(128)

train_x = train_x.reshape(train_x.shape[0], -1)
train_x = test_x.reshape(test_x.shape[0], -1)
train_t = train_t[:,1]
test_t = test_t[:,1]

clf = svm.SVC()
clf.fit(train_x, train_t)
train_predict = clf.predict(train_x)
corrects = (train_predict==train_t).astype(np.float32).sum()
accuracy = corrects/train_x.shape[0]

print(accuracy)
