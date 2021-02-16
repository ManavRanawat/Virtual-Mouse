# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:55:43 2021

@author: Manav Ranawat
"""

import cv2
import numpy as np
from collections import deque 


cap = cv2.VideoCapture(0)
bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
kernel = np.ones((5,5),np.uint8)


while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hand_cascade = cv2.CascadeClassifier('hand.xml')
    # hands = hand_cascade.detectMultiScale(frame,1.1,3)
    
    # print(hands)
    # cv2.rectangle()
    
    
    hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 55, 40], dtype = "uint8")
    upper = np.array([35, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    skinRegionHSV =  cv2.erode(skinRegionHSV, kernel, iterations = 2)
    skinRegionHSV = cv2.dilate(skinRegionHSV, kernel, iterations = 2)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    # fgmask = bgSubtractor.apply(blurred,learningRate=0)
    
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)

    
    
    # lower = np.array([0, 68, 50], dtype = "uint8")
    # upper = np.array([40, 255, 255], dtype = "uint8")
    # skinMask = cv2.inRange(frame,lower,upper)
    
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    # skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    
    cv2.imshow('mask',thresh)
    cv2.imshow('frame',frame)



    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()
cap.release()