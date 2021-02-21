# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:55:43 2021

"""

import cv2
import imutils
import numpy as np
from collections import deque 


cap = cv2.VideoCapture(0)
# bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
kernel = np.ones((5,5),np.uint8)
cnt = 0

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
    
    


    
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    # for (x, y, w, h) in faces:
    
    if len(faces)>0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        thresh[y:y+h,x:x+w] = np.zeros((h, w), np.float64)
    else:
        continue
    



    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) <= 0:
        continue
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

    
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(frame, [c], -1, (255, 0, 255), 2)
    cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
    cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
    cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

    
    hull = cv2.convexHull(c)
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)

    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)


    # if defects is not None:
    #     cnt = 0
    #     for i in range(defects.shape[0]):
    #         s, e, f, d = defects[i][0]
    #         start = tuple(cnts[s][0])
    #         end = tuple(cnts[e][0])
    #         far = tuple(cnts[f][0])
    #         a1 = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    #         b1 = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    #         c1 = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    #         angle = np.arccos((b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1))  #      cosine theorem
    #         if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
    #             cnt += 1
    #             cv.circle(frame, far, 4, [0, 0, 255], -1)
    #     if cnt > 0:
    #         cnt = cnt+1
    #     cv.putText(frame, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # put text and highlight the center
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    cv2.imshow('mask',thresh)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        # print(defects)
        break


    



cv2.destroyAllWindows()
cap.release()