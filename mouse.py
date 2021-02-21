# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:55:43 2021

"""

import cv2
import imutils
import numpy as np
from collections import deque 
from sklearn.metrics import pairwise
import pyautogui as gui


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
kernel = np.ones((5,5),np.uint8)
cnt = 0
flag=True
d = deque(maxlen=20)


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
    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    if M["m00"]==0:
        continue
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # put text and highlight the center
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    d.appendleft((cX,cY))
    if len(d)>1:
        gui.move(5*(d[0][0]-d[1][0]),5*(d[0][1]-d[1][1]))

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

    # cX=(extLeft[0]+extRight[0])//2
    # cY=(extTop[1]+extBot[1])//2

    
    hull = cv2.convexHull(c)
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)

    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)
    # cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
    
    # if defects is not None:
    #     cnt = 0
    #     for i in range(defects.shape[0]):
    #         s, e, f, d = defects[i][0]
    #         start = tuple(c[s][0])
            
    #         # if flag==True:
    #         #     print(cnts)
    #         #     flag=False
    #         end = tuple(c[e][0])
    #         far = tuple(c[f][0])
    #         a1 = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    #         b1 = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    #         c1 = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    #         angle = np.arccos((b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1))  #      cosine theorem
    #         if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
    #             cnt += 1
    #             cv2.circle(frame, far, 4, [0, 0, 255], -1)
    #     if cnt > 0:
    #         cnt = cnt+1
    #     cv2.putText(frame, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)

    dist=pairwise.euclidean_distances([extLeft,extRight,extBot,extTop],[[cX,cY]])[0]
    radi=int(0.80*dist)

    t2 = thresh.copy()
    
    circular_roi=np.zeros_like(thresh,dtype='uint8')
    cv2.circle(circular_roi,(cX,cY),radi,255,8)
    wighted=cv2.addWeighted(thresh.copy(),0.6,circular_roi,0.4,2)

    mask=cv2.bitwise_and(t2,t2,mask=circular_roi)
    #mask
    con,hie=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count=0
    circumfrence=2*np.pi*radi
    for cnt in con:
        (m_x,m_y,m_w,m_h)=cv2.boundingRect(cnt)
        out_wrist_range=(cY+(cY*0.25))>(m_y+m_h)
        limit_pts=(circumfrence*0.25)>cnt.shape[0]
        if limit_pts and out_wrist_range:
            count+=1


    cv2.putText(frame,'count: '+str(count),(460,70),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,250,0),thickness=4)
    # cv2.rectangle(frame,(x,y),(x+w,y+h),255,3)

    cv2.imshow('weight',wighted)
    cv2.imshow('mask',thresh)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        # print(defects)
        break


    



cv2.destroyAllWindows()
cap.release()