import cv2
import mediapipe as mp
import HandTrackingModule as htm
import os
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.HandDetector(detectCon=0.75)
folderPath = "Header"
imagePath = os.listdir(folderPath)
overlayList = []
drawColour = 0
xp, yp = 0, 0
#ImageCanvas = np.zeros((1280,720,3), np.uint64)
for image in imagePath:
    im = cv2.imread(f'{folderPath}/{image}')
    overlayList.append(im)
header = overlayList[0]
imgCanvas = np.zeros((720,1280,3),np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        fingers = detector.fingersUp()
        #print(fingers)
        if y1<130:
            if 150<x1<250:
                drawColour = (255,0,255)
                header = overlayList[0]
            elif 350<x1<550:
                drawColour = (255, 0, 0)
                header = overlayList[1]
            elif 600<x1<750:
                drawColour = (0, 255, 0)
                header = overlayList[2]
            elif 850<x1<1000:
                drawColour = (0, 0, 0)
                header = overlayList[3]
        if fingers[1] and fingers[2]==False:
            print("Drawing Mode")
            cv2.circle(img, (x1,y1), 15, drawColour, cv2.FILLED)
            if xp==0 and yp==0:
                xp, yp = x1, y1
            if drawColour==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColour, 50)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, 50)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColour,15)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, 15)
            xp, yp = x1, y1
        if fingers[1] and fingers[2]:
            print("selection Mode")
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColour, cv2.FILLED)


    header = cv2.resize(header,(1280,130))
    img[0:130, 0:1280] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("image", img)
    #cv2.imshow("image canvas", imgCanvas)
    cv2.waitKey(1)

