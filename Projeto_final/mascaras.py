import numpy as np
import cv2
import time

video = "videocorreto.mp4"

gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
knn = cv2.createBackgroundSubtractorKNN()
cnt = cv2.bgsegm.createBackgroundSubtractorCNT()

cap = cv2.VideoCapture(video) 


while (cap.isOpened):
    true, frame = cap.read()

    if not true:
      print('Frames acabaram!')
      break

    frame = cv2.resize(frame, (0, 0), fx=0.48, fy=0.48)

    mask_gmg = gmg.apply(frame)
    mask_mog = mog.apply(frame)
    mask_mog2 = mog2.apply(frame)
    mask_knn = knn.apply(frame)
    mask_cnt = cnt.apply(frame)


    cv2.imshow('Video', frame)
    cv2.imshow('gmg', mask_gmg)
    cv2.imshow('mog', mask_mog)
    cv2.imshow('mog2', mask_mog2)
    cv2.imshow('knn', mask_knn)
    cv2.imshow('cnt', mask_cnt)

    time.sleep(0.07) 

    if cv2.waitKey(1) == 27: 
        break