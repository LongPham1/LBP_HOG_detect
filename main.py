# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 06:42:28 2023

@author: ADMIN
"""

import cv2
import os
import sys
import  time
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shutil
import glob
from skimage import data, color, feature, transform
import skimage.data
from joblib import Parallel, delayed
import imutils

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# Mặt
cap = cv2.VideoCapture(0)  
# cap = cv2.VideoCapture("D:/tai xuong/20210628_Face_Detection_with_Dlib_using_HOG_and_Linear_SVM/20210628_Face_Detection_with_Dlib_using_HOG_and_Linear_SVM/input/video1.mp4") 

dataset_folder = "D:/tai xuong/"
model = joblib.load(dataset_folder + "best_model (2) .pkl")      

fps = 0 
frame_counter = 0
start_time = time.time()

def kiemtra(test_image):
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) 
    indices, patches = zip(*sliding_window(test_image))

    npoint = 8
    rad = 1.0
    patches_hog = []
    
    for im in patches:
        lbp_feature = feature.local_binary_pattern(im, npoint, rad)
        lbp_feature = np.histogram(lbp_feature.ravel(), bins=np.arange(0, npoint + 3), range=(0, npoint + 2))[0]
        hog_feature = feature.hog(im)
        concatenated_feature = np.concatenate((hog_feature, lbp_feature))
        # result = model.predict_proba([concatenated_feature])  
        patches_hog.append(concatenated_feature)

    labels = model.predict(patches_hog)
    indices = np.array(indices)
    
    for i, j in indices[labels == 1]:   
        k = cv2.rectangle(img, (j, i), (j + 45, i + 45), (0, 255, 0), 2)     
        # cv2.putText(img, '{}: {:.2f}%'.format("face", (result[0][0]) * 100 - 1), (j, i - 5),     
        # cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 255), 1)   
     


def sliding_window(img, patch_size=(45,45),
                   istep=15, jstep=15, scale=1):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


while True:
    ret, img = cap.read()
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    
    if img is not None:
        img = cv2.resize(img, (155, 155))  
        kiemtra(img)
        img = imutils.resize(img, width=410)   
            
    cv2.putText(img, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),2)
    cv2.imshow('Capture Faces', img)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""

cap.release()
cv2.destroyAllWindows()


img = cv2.imread("D:/tai xuong/images.jpg")
img = cv2.resize(img, (150, 150))
kiemtra(img)
cv2.imshow('Capture Faces', img)

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()  
"""
