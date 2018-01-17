import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from mne.io import concatenate_raws
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.eegnet import EEGNet
from braindecode.models.deep4 import Deep4Net
from torch import nn
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_for_set
import os, sys
os.chdir('/home/seigyo/Documents/pytorch/brain_decoder')
from load_data import get_data, get_data_multi

X, y = get_data(id=10, event_code=[5,6,9,10,13,14], filter=None, t=[-0.5, 4])
train_X, train_y = X[:60], y[:60]
test_X, test_y = X[60:], y[60:]

# train_X, train_y = get_data_multi(sub_id_range=[1, 51], event_code=[5,9,13], filter=None, t=[1, 4.1])
# test_X, test_y = get_data_multi(sub_id_range=[51, 55], event_code=[5,9,13], filter=None, t=[1, 4.1])

train_set = SignalAndTarget(train_X, y=train_y)
test_set = SignalAndTarget(test_X, y=test_y)

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = train_set.X.shape[2]
n_classes = 2
in_chans = train_set.X.shape[1]

# final_conv_length determines the size of the receptive field of the ConvNet
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=input_time_length,
                        final_conv_length='auto').create_network()
if cuda:
    model.cuda()



optimizer = optim.Adam(model.parameters())

rng = RandomState((2017,6,30))
for i_epoch in range(30):
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                            batch_size=30)
    # Set model to training mode
    model.train()
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X
        batch_X = train_set.X[i_trials][:,:,:,None]
        batch_y = train_set.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    for setname, dataset in (('Train', train_set), ('Test', test_set)):
        # Here, we will use the entire dataset at once, which is still possible
        # for such smaller datasets. Otherwise we would have to use batches.
        net_in = np_to_var(dataset.X[:,:,:,None])
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(dataset.y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in)
        loss = F.nll_loss(outputs, net_target)
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))
