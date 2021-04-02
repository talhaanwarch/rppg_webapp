
import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
from .model import Attention_mask, MTTS_CAN
import h5py
#import matplotlib.pyplot as plt
from scipy.signal import butter
from .inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import find_peaks
main_path=os.path.dirname(os.path.dirname(__file__))

def hear_rate(peaklist,fs):
    RR_list = []
    cnt = 0
    
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    
    bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
    print ("Average Heart Beat is: %.01f" %bpm) #Round off to 1 decimal and print
    return np.round(bpm,3)


def predict_vitals(vid,d=15):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = main_path+'/py_templates'+'/mtts_can.hdf5'
    print(model_checkpoint)
    batch_size = 100
    sample_data_path = vid
    dXsub,fs = preprocess_raw_video(sample_data_path, dim=36)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
    print('+++++++++++++++++++++++++++++++++++++++++')
    print(yptest)
    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(2, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    
    ########## calculate peaks ####################
    peaks, _ = find_peaks(pulse_pred, distance=d)
    return hear_rate(peaks,fs)


