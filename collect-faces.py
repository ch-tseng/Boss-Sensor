#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import sys
import time
import datetime

savePath = "datasets\\collected"
cam_id = 0
face_size = (47, 62)
monitor_winSize = (640, 480)
cam_resolution = (1080,960)

face_cascade = cv2.CascadeClassifier('objects\\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('objects\\lbpcascade_frontalface.xml')
	
camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])
	
while True:
    (grabbed, img) = camera.read()    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= 1.3,
        minNeighbors=10,
        minSize=face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    i = 0
    for (x,y,w,h) in faces:
	
        if(w>face_size[0] and h>face_size[1]):
            roi_color = img[y:y+h, x:x+w]
            now=datetime.datetime.now()
            faceName = '%s_%s_%s_%s_%s_%s_%s.jpg' % (now.year, now.month, now.day, now.hour, now.minute, now.second, i)
            cv2.imwrite(savePath+"\\" + faceName, roi_color)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    r = monitor_winSize[1] / img.shape[1]
    dim = (monitor_winSize[0], int(img.shape[0] * r))
    img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Frame", img2)
    key = cv2.waitKey(1)
    
