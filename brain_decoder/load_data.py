import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from mne.io import concatenate_raws

'''
sub_id can be range of 1-109.
runs
1: baseline of eyes open
2: baseline of eyes close
3,7,11: motor execution left vs right hands
4,8,12: motor imagery left vs right hands
5,9,13: motor execution hands vs feet
6,10,14: motor imagery hands vs feet
'''

def get_data_one_class(id=1, event_code=[5,6,9,10,13,14], filter=[0.5, 36], t=[1, 4.1], classid=2):
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = id
    event_codes = event_code

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

    # Load each of the files
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto', verbose='WARNING')
             for path in physionet_paths]

    # Concatenate them
    raw = concatenate_raws(parts)

    # bandpass filter
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    # Find the events in this dataset
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)
    # change time length
    # epochs_train = epochs.copy().crop(tmin=1., tmax=2.)


    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_data_one_class_multi(sub_id_range=[1, 50], event_code=[5,6,9,10,13,14], t=[1, 4.1], filter=[0.5,36]):
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
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y

def get_data(id=1, event_code=[5,6,9,10,13,14], filter=[0.5, 36], t=[1, 4.1]):
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = id
    event_codes = event_code

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

    # Load each of the files
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto', verbose='WARNING')
             for path in physionet_paths]

    # Concatenate them
    raw = concatenate_raws(parts)

    # bandpass filter
    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    # Find the events in this dataset
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
                    baseline=None, preload=True)
    # change time length
    # epochs_train = epochs.copy().crop(tmin=1., tmax=2.)


    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    return X, y


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

def get_crops(id=1, event_code=[5,6,9,10,13,14], filter=[0.5, 36], t=[0, 4.0],
               time_window=1.0, time_step=0.5):
    subject_id = id
    event_codes = event_code
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto', verbose='WARNING')
             for path in physionet_paths]

    raw = concatenate_raws(parts)

    if filter != None:
        raw.filter(filter[0], filter[1], fir_design='firwin', skip_by_annotation='edge')
    else:
        pass

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
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

    # epochs_list = []
    # start = t[0]
    # while True:
    #     end = start + time_window
    #     if end > t[1]:
    #         break
    #     epochs_list.append(epochs.copy().crop(tmin=start, tmax=end))
    #     # print('get_time {} to {}'.format(start, end))
    #     start += time_step

    # # for i in range(int(160 * (t[1]-t[0]-time_window) / (160*time_step))):
    # #     epochs_train = epochs.copy().crop(tmin=i * time_window/160 * (160*time_step), tmax=i * time_window/160 * (160*time_step) + time_window)
    # #     epochs_list.append(epochs_train)

    # data_list = []
    # label_list = []
    # for epoch in epochs_list:
    #     X = (epoch.get_data() * 1e6).astype(np.float32)
    #     y = (epoch.events[:,2] - 2).astype(np.int64)
    #     data_list.append(X)
    #     label_list.append(y)
    # data_array = np.array(data_list)
    # label_array = np.array(label_list)
    # data_array2 = data_array.reshape(-1, data_array.shape[-2], data_array.shape[-1])
    # label_array2 = label_array.reshape(-1)

    return x, y, raw, epochs

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


    # epochs_list = []
    # start = t[0]
    # while True:
    #     end = start + time_window
    #     if end > t[1]:
    #         break
    #     epochs_list.append(epochs.copy().crop(tmin=start, tmax=end))
    #     # print('get_time {} to {}'.format(start, end))
    #     start += time_step
    # # epochs_list = []
    # # for i in range(int(160 * (t[1]-t[0]-time_window) / (160*time_step))):
    # #     epochs_train = epochs.copy().crop(tmin=i * time_window/160 * (160*time_step), tmax=i * time_window/160 * (160*time_step) + time_window)
    # #     epochs_list.append(epochs_train)
    # data_list = []
    # label_list = []
    # for epoch in epochs_list:
    #     X = (epoch.get_data() * 1e6).astype(np.float32)
    #     y = (epoch.events[:,2] - 2).astype(np.int64)
    #     data_list.append(X)
    #     label_list.append(y)
    # data_array = np.array(data_list)
    # label_array = np.array(label_list)
    # data_array2 = data_array.reshape(-1, data_array.shape[-2], data_array.shape[-1])
    # label_array2 = label_array.reshape(-1)

    return x, y, raw, epochs

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
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched = mne.Epochs(raw, events, classid, tmin=t[0], tmax=t[1], proj=False, picks=eeg_channel_inds,
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


    # epochs_list = []
    # start = t[0]
    # while True:
    #     end = start + time_window
    #     if end > t[1]:
    #         break
    #     epochs_list.append(epochs.copy().crop(tmin=start, tmax=end))
    #     # print('get_time {} to {}'.format(start, end))
    #     start += time_step
    # # epochs_list = []
    # # for i in range(int(160 * (t[1]-t[0]-time_window) / (160*time_step))):
    # #     epochs_train = epochs.copy().crop(tmin=i * time_window/160 * (160*time_step), tmax=i * time_window/160 * (160*time_step) + time_window)
    # #     epochs_list.append(epochs_train)
    # data_list = []
    # label_list = []
    # for epoch in epochs_list:
    #     X = (epoch.get_data() * 1e6).astype(np.float32)
    #     y = (epoch.events[:,2] - 2).astype(np.int64)
    #     data_list.append(X)
    #     label_list.append(y)
    # data_array = np.array(data_list)
    # label_array = np.array(label_list)
    # data_array2 = data_array.reshape(-1, data_array.shape[-2], data_array.shape[-1])
    # label_array2 = label_array.reshape(-1)

    return x, y, raw, epochs