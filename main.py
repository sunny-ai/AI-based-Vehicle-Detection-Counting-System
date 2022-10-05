 # -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 02:10:56 2022

@author: Lite Computer
"""

import cv2
import numpy as np 
import serial
import time

arduino = serial.Serial(port='COM3', baudrate=115200, timeout = 2) #arduino port
time.sleep(2)

#path of the video file
cap = cv2.VideoCapture('C:/Users/Hp/Desktop/Object_flagging/video.mp4')

_, prev = cap.read()
prev = cv2.flip(prev, 1)
_, new = cap.read()
new = cv2.flip(new, 1)

thres = 0.45

cap.set(3,640)
cap.set(4,320)
cap.set(10,70)

count = 0
classNames= []
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'frozen_inference_graph.pb'
weightsPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def detection():
    for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                        cv2.putText(prev,classNames[classId-1].upper(),(box[0]+10,box[0]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

                       # CL=(classNames[classId-1].upper())
                        count = np.count_nonzero(classIds == 3)
                        
                       # print(np.count_nonzero(classIds == 3))
                        print(count)
                        #print(classIds)
                        
                        cv2.putText(prev,str(count),(200,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        

                        if count>7:
                            #Green ligh
                            cv2.circle(prev, (100,40), 30, (0,0,255), -1)
                        else:
                            #Red light
                            cv2.circle(prev, (100,40), 30, (0,128,0), -1)

while True:
    classIds, confs, bbox = net.detect(prev,confThreshold=thres)
    #print(classIds,bbox)                    
    diff = cv2.absdiff(prev, new)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.blur(diff, (5,5))
    _,thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    threh = cv2.dilate(thresh, None, 3)
    thresh = cv2.erode(thresh, np.ones((4,4)), 1)
    contor,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(classIds) != 0:
        detection()	
	
    cv2.imshow("orig", prev)
	
    prev = new
    _, new = cap.read()
    new = cv2.flip(new, 1)

    if cv2.waitKey(1) == 27:
        break
    
    
if count>=7:
    arduino.write('1')
    print("LED")

cap.release()
cv2.destroyAllWindows()


    
