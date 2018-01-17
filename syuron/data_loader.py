import mne
from mne.io import concatenate_raws
import torch
import numpy as np
from data_utils import get_crops_multi, get_crops_multi_one_class
from data_utils import get_data_one_class_multi, get_data_multi


def make_class(subject_id=[1,10], problem='hf', bpfilter = [0.5, 45]):
    if problem == "hf":
        X, y = get_data_multi(sub_id_range=subject_id,
                              event_code=[6,10,14],
                              t=[0, 4.0],
                              filter=bpfilter)
  
    if problem == "lr":
        X, y = get_data_multi(sub_id_range=subject_id,
                              event_code=[4,8,12],
                              t=[0, 4.0],
                              filter=bpfilter)
  
    if problem == "lh":
        Xl, yl = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xh, yh = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        X = np.vstack((Xl,Xh))
        y = np.hstack((yl,yh+1))
  
    if problem == "rh":
        Xr, yr = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xh, yh = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        X = np.vstack((Xr,Xh))
        y = np.hstack((yr-1,yh+1))
  
    if problem == "lf":
        Xl, yl = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xf, yf = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        X = np.vstack((Xl,Xf))
        y = np.hstack((yl,yf))
  
    if problem == "rf":
        Xr, yr = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xf, yf = get_data_one_class_multi(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        X = np.vstack((Xr,Xf))
        y = np.hstack((yr-1,yf))
    return X, y

def make_class_crop(subject_id=[1,10], problem='hf', bpfilter = [0.5, 45],
                    time_window=1.0, time_step=0.5):
 
    if problem == "hf":
        X, y = get_crops_multi(sub_id_range=subject_id,
                              event_code=[6,10,14],
                              t=[0, 4.0],
                              filter=bpfilter,
                              time_window=time_window,
                              time_step=time_step)
  
    if problem == "lr":
        X, y = get_crops_multi(sub_id_range=subject_id,
                              event_code=[4,8,12],
                              t=[0, 4.0],
                              filter=bpfilter,
                              time_window=time_window,
                              time_step=time_step)
  
    if problem == "lh":
        Xl, yl = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        Xh, yh = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xl,Xh))
        y = np.hstack((yl,yh+1))
  
    if problem == "rh":
        Xr, yr = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)
        Xh, yh = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xr,Xh))
        y = np.hstack((yr-1,yh+1))
  
    if problem == "lf":
        Xl, yl = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=2)
        Xf, yf = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)

        X = np.vstack((Xl,Xf))
        y = np.hstack((yl,yf))
  
    if problem == "rf":
        Xr, yr = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[4,8,12],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3)
        Xf, yf = get_crops_multi_one_class(sub_id_range=subject_id,
                                          event_code=[6,10,14],
                                          filter=bpfilter,
                                          t=[0, 4],
                                          classid=3,
                                          time_window=time_window,
                                          time_step=time_step)
        X = np.vstack((Xr,Xf))
        y = np.hstack((yr-1,yf))
    return X, y


def make_4class(subject_id=[1,10], bpfilter = [0.5, 45]):
    X1, y1 = get_data_multi(sub_id_range=subject_id,
                            event_code=[6,10,14],
                            t=[0, 4.0],
                            filter=bpfilter)
  
    X2, y2 = get_data_multi(sub_id_range=subject_id,
                            event_code=[4,8,12],
                            t=[0, 4.0],
                            filter=bpfilter)

    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2+2))
    return X, y

def make_4class_crops(subject_id=[1,10], bpfilter = [0.5, 45], time_window=1, time_step=0.5):
    X1, y1 = get_crops_multi(sub_id_range=subject_id,
                            event_code=[6,10,14],
                            t=[0, 4.0],
                            filter=bpfilter,
                            time_window=time_window,
                            time_step=time_step)
  
    X2, y2 = get_crops_multi(sub_id_range=subject_id,
                            event_code=[4,8,12],
                            t=[0, 4.0],
                            filter=bpfilter,
                            time_window=time_window,
                            time_step=time_step)

    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2+2))
    return X, y

def main():
    np.set_printoptions(threshold=np.inf)
    num = input("module_test_numbet 1~4:")
    num = int(num)
    if num == 1:
        X, y = make_class(subject_id=[1,10], problem='lf', bpfilter = [0.5, 45])
        print(X.shape)
        print(y)

    elif num == 2:
        X, y = make_class_crop(subject_id=[1,10], problem='lf', bpfilter = [0.5, 45],
                            time_window=1.0, time_step=0.5)
        print(X.shape)
        print(y)

    elif num == 3:
        X, y = make_4class(subject_id=[1,10], bpfilter = [0.5, 45])
        print(X.shape)
        print(y)

    elif num == 4:   
        X, y = make_4class_crops(subject_id=[1,10], bpfilter = [0.5, 45],
                                time_window=1, time_step=0.5)
        print(X.shape)
        print(y)

    else:
        print("no module test")
if __name__ == '__main__':
    main()