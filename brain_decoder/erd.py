import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
%matplotlib inline

def center_cmap(cmap, vmin, vmax):
    """Center given colormap (ranging from vmin to vmax) at value 0.

    Note that eventually this could also be achieved by re-normalizing a given
    colormap by subclassing matplotlib.colors.Normalize as described here:
    https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    """  # noqa: E501
    vzero = abs(vmin) / (vmax - vmin)
    index_old = np.linspace(0, 1, cmap.N)
    index_new = np.hstack([np.linspace(0, vzero, cmap.N // 2, endpoint=False),
                           np.linspace(vzero, 1, cmap.N // 2)])
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}
    for old, new in zip(index_old, index_new):
        r, g, b, a = cmap(old)
        cdict["red"].append((new, r, r))
        cdict["green"].append((new, g, g))
        cdict["blue"].append((new, b, b))
        cdict["alpha"].append((new, a, a))
    return LinearSegmentedColormap("erds", cdict)


# load and preprocess data ####################################################
subject = 20  # use data from subject 1
runs = [3, 7, 11]  # use only hand and feet motor imagery runs


fnames = mne.datasets.eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(f, preload=True, stim_channel='auto') for f in fnames]
raw = concatenate_raws(raws)

raw.rename_channels(lambda x: x.strip('.'))  # remove dots from channel names

events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

# epoch data ##################################################################
tmin, tmax = -1, 4  # define epochs around events (in s)
event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white

for event in event_ids:
    power = tfr_multitaper(epochs[event], freqs=freqs, n_cycles=n_cycles,
                           use_fft=True, return_itc=False, decim=2)
    power.crop(tmin, tmax)

    fig, ax = plt.subplots(1, 4, figsize=(12, 4),
                           gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for i in range(3):
        power.plot([i], baseline=[-1, 0], mode="percent", vmin=vmin, vmax=vmax,
                   cmap=(cmap, False), axes=ax[i], colorbar=False, show=False)
        ax[i].set_title(epochs.ch_names[i], fontsize=10)
        ax[i].axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if i > 0:
            ax[i].set_ylabel("")
            ax[i].set_yticklabels("")
    fig.colorbar(ax[0].collections[0], cax=ax[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
