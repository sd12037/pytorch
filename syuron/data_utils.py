import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from mne.io import concatenate_raws



def get_data_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[1, 4.1], filter=[0.5,36]):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]
    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=picks,
                    baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_data_one_class_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[1, 4.1], filter=[0.5,36], classid=2):
    physionet_paths = [mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]
    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    
    epoched = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_crops_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[0, 4.0], filter=[0.5,36],
                    time_window=1.0, time_step=0.5):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]

    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=picks,
                    baseline=None, preload=True)

    ### startからendまでcrop,初期配列
    start = t[0]
    end = start + time_window
    this_epoch = epochs.copy().crop(tmin=start, tmax=end)
    x = (this_epoch.get_data()*1e6).astype(np.float32)
    y = (this_epoch.events[:,2]-2).astype(np.int64)  
    print('get_time {} to {}'.format(start, end))

    ### 繰り返しcropデータを連結
    while True:
        start += time_step
        end = start + time_window
        if end > t[1]:
            break
        this_epoch = epochs.copy().crop(tmin=start, tmax=end)
        x = np.vstack((x, (this_epoch.get_data()*1e6).astype(np.float32)))
        y = np.hstack((y, (this_epoch.events[:,2]-2).astype(np.int64)))   
        print('get_time {} to {}'.format(start, end))

    return x, y

def get_crops_multi_one_class(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[0, 4.0], filter=[0.5,36],
                              time_window=1.0, time_step=0.5, classid=2):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,event_code) for sub_id in range(sub_id_range[0],sub_id_range[1])]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]

    raw = concatenate_raws(parts)
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)

    ### startからendまでcrop,初期配列
    start = t[0]
    end = start + time_window
    this_epoch = epochs.copy().crop(tmin=start, tmax=end)
    x = (this_epoch.get_data()*1e6).astype(np.float32)
    y = (this_epoch.events[:,2]-2).astype(np.int64)  
    print('get_time {} to {}'.format(start, end))

    ### 繰り返しcropデータを連結
    while True:
        start += time_step
        end = start + time_window
        if end > t[1]:
            break
        this_epoch = epochs.copy().crop(tmin=start, tmax=end)
        x = np.vstack((x, (this_epoch.get_data()*1e6).astype(np.float32)))
        y = np.hstack((y, (this_epoch.events[:,2]-2).astype(np.int64)))   
        print('get_time {} to {}'.format(start, end))

    return x, y