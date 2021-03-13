# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:31:03 2021
82,86,86,91,83|||82,92,90,88,81
@author: TAC
"""

import cv2
import numpy as np
from scipy import signal
import os
main_path=os.path.dirname(os.path.dirname(__file__))
from time import time
def fourier_analysis(array,fps):

    def remove_outliers(arr, thr):
        return next(f[0] for f in enumerate(arr) if f[1] > thr)
    MinFreq = 45   # bpm
    MaxFreq = 100  # bpm
    freqs, psd = signal.periodogram(array, fs=fps, window=None, \
                                    detrend='constant', return_onesided=True, \
                                        scaling='density')
    min_idx = remove_outliers(freqs, MinFreq/60.0) - 1
    max_idx = remove_outliers(freqs, MaxFreq/60.0) + 1
    hr_estimated = freqs[ min_idx + np.argmax(psd[min_idx : max_idx]) ]
    return hr_estimated
def filter(array,low,high):
    fs=fps #sampling frequency
    nyq = 0.5 * fs#nyquist theroem
    low = low / nyq #low cutoff
    high = high / nyq#high cutff
    b, a = signal.butter(3, [low,high], 'band') #order
    return signal.filtfilt(b,a,array)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

## Step 1: select pathces
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

def patch_extract(path):
    global fps
    cap = cv2.VideoCapture(main_path+path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps',fps)
    patches_head,patches_cheeck,patchesr=[],[],[]
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")

    start=time()  
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        
        
        #resize frame https://stackoverflow.com/a/44659589/11170350
        (h, w) = frame.shape[:2]
        if h>w:
            height=720
            r = height / float(h)
            dim = (int(w * r), height)
            frame=cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        else:
            width=720
            r = width / float(w)
            dim = (width, int(h * r))
            frame=cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x+w//2-50,y+10),(x+w//2+50,y+h//4-10),(255,0,0),2)
            img = cv2.rectangle(img,(x+w//2-50,y+h//2+40),(x+w//2+50,y+h//2-15),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            forhead = img[y+10:y+h//4-10, x+w//2-50:x+w//2+50]
            patches_cheeck.append(img[y+h//2-15:y+h//2+40, x+w//2-50:x+w//2+50])
            patches_head.append(forhead)
      else: 
          break
    cap.release()
    
    min_width=min([len(i) for i in patches_head])
    patchesfeq=[i[0:min_width] for i in patches_head]

    patches_cheeck_eq=[i[0:min_width] for i in patches_cheeck]

    patch_arr=np.mean([patchesfeq,patches_cheeck_eq],axis=0)

    patch_arr=patch_arr[0:len(patch_arr)-len(patch_arr)%100,:,:,:]
    print(time()-start)
    return patch_arr




#make all pathces of equal size

def heart_rate(patch_arr):
    o=5
    hr_fourier=[]
    for p in range(0,len(patch_arr),len(patch_arr)//o):
        patch_select=patch_arr[p:p+len(patch_arr)//o]
        r,g,b=[],[],[]
        for i in range(len(patch_select)):
            r.append(np.mean(patch_select[i,:,:,2]))
            g.append(np.mean(patch_select[i,:,:,1]))
            b.append(np.mean(patch_select[i,:,:,0]))
        
        r,g,b=np.array(r),np.array(g),np.array(b)
        
        RGB = np.transpose(np.array([r,g,b]))

        l=30
        
        num_frames=len(patch_select)
        H = np.zeros(num_frames)
        
        for n in range(num_frames-l):
                m=n-l+1
                C = RGB[m:n,:].T
                if m>=0:     
                    mean_color = np.mean(C, axis=1)        
                    diag_mean_color = np.diag(mean_color)
                    diag_mean_color_inv = np.linalg.inv(diag_mean_color)
                    Cn = np.matmul(diag_mean_color_inv,C)
                    projection_matrix = np.array([[0,1,-1],[-2,1,1]])            
                    S = np.matmul(projection_matrix,Cn)            
                    S[0,:] = filter(S[0,:], 0.5, 4.0)
                    S[1,:] = filter(S[1,:], 0.5, 4.0)            
                    std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
                    h = np.matmul(std,S)
                    
                    H[m:n] = H[m:n] + (h-np.mean(h))/np.std(h)

        hr_fourier_pos = fourier_analysis(H, fps)
        print(hr_fourier_pos*60)
        hr_fourier.append(hr_fourier_pos*60)

    print('calculated mean', np.mean(hr_fourier))
    print('calculated std', np.std(hr_fourier))
    return hr_fourier

# print('ground mean', np.mean([82,92,90,88,81]))
# print('ground std', np.std([82,86,86,91,83]))







