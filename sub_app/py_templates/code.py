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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('sub_app/media/video/output.mp4',fourcc, 20.0, (720,405))
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
        faces = face_cascade.detectMultiScale(gray,1.3, 5)
        cl=60#cheeck length
        fh=6#forhead from top to
        for (x,y,w,h) in faces:
            #img = cv2.rectangle(frame,(x+w//2-cl,y),(x+w//2+cl,y+(h//fh)),(255,0,0),2)#forehead
            img = cv2.rectangle(frame,(x+w//2-cl,y+h//2+40),(x+w//2+cl,y+h//2-15),(255,0,0),2)#cheeks
            #roi_gray = gray[y:y+h, x:x+w]
            #patches_head.append(img[y:y+(h//fh), x+w//2-cl:x+w//2+cl])
            patches_cheeck.append(frame[y+h//2-15:y+h//2+40, x+w//2-cl:x+w//2+cl])


        out.write(frame)

      else: 
          break
    cap.release()
    out.release()

    #head #remove frame have less height than average 
    #patch_len=np.mean([len(i) for i in patches_head])
    #patches_head=[i for i in patches_head if len(i) > patch_len-5]

	#cheeck #remove frame have less length than average 
    patch_len=np.mean([i.shape[1] for i in patches_cheeck])
    patches_cheeck=[i for i in patches_cheeck if len(i) < patch_len-5]

	#now make both equal length
    # min_len=min(len(patches_head),len(patches_cheeck))
    # patches_head=patches_head[0:min_len]
    # patches_cheeck=patches_cheeck[0:min_len]
    # patch_arr=np.mean([patches_head,patches_cheeck],axis=0)
    patches_cheeck=np.array(patches_cheeck)
    patches_cheeck=patches_cheeck[0:len(patches_cheeck)-len(patches_cheeck)%100,:,:,:]
    return patches_cheeck




#make all pathces of equal size

def heart_rate(patch_arr):
    if len(patch_arr)<200:
        o=1
    elif len(patch_arr)>200 and len(patch_arr)<600:
        o=2
    else:
        o=3
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
        hr_fourier.append(hr_fourier_pos*60)

    print('calculated mean', np.mean(hr_fourier))
    print('calculated std', np.std(hr_fourier))
    return hr_fourier

# print('ground mean', np.mean([82,92,90,88,81]))
# print('ground std', np.std([82,86,86,91,83]))







