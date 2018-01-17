import os, sys
os.chdir('/home/hikaru/Documents/pytorch/brain_decoder')
import numpy as np
import scipy
from scipy.io import loadmat
from eeg_cnn_lib import *
# from eeglearn.utils import reformatInput
# Load electrode locations
print('Loading data...')
locs = scipy.io.loadmat('../Sample_data/Neuroscan_locs_orig.mat')
locs_3d = locs['A']
locs_2d = []
# Convert to 2D
for e in locs_3d:
    locs_2d.append(azim_proj(e))

feats = scipy.io.loadmat('../Sample_data/FeatureMat_timeWin.mat')['features']
subj_nums = np.squeeze(scipy.io.loadmat('../Sample_data/trials_subNums.mat')['subjectNum'])
# Leave-Subject-Out cross validation
fold_pairs = []
for i in np.unique(subj_nums):
    ts = subj_nums == i
    tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
    ts = np.squeeze(np.nonzero(ts))
    np.random.shuffle(tr)  # Shuffle indices
    np.random.shuffle(ts)
    fold_pairs.append((tr, ts))

# CNN Mode
print('Generating images...')
# Find the average response over time windows
av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(int(feats.shape[1] / 192))])
av_feats = av_feats / (feats.shape[1] / 192)
images = gen_images(np.array(locs_2d),
                              av_feats,
                              32, normalize=False)
print(images.shape)
feats.shape[1]
# train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'cnn')

images_timewin = np.array([gen_images(np.array(locs_2d),
                                               feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
                                    range(int(feats.shape[1] / 192))
                                    ])
print(images_timewin.shape)
# train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'mix')
