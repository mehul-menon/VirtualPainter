import mediapipe as mp
import time
import cv2




class HandDetector():

    def __init__(self, mode=False, maxhand=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode#define init function for class and specify arguments of Hands function
        self.maxhand = maxhand
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.tip_id = [4, 8, 12, 16, 20]
        # ctrl and click for parameters. static image mode at false means the tracking is done only with minimum confidence level
        # this is faster than when it is true. brackets are empty to use default values

        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # to check use print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:#checking if landmarks are detected
          for x in self.results.multi_hand_landmarks:#looping over landmarks

            if draw:
                self.mpDraw.draw_landmarks(img,x,self.mpHands.HAND_CONNECTIONS)#3rd parameter draws connecting lines
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
              h, w, c = img.shape
              cx, cy = int(lm.x*w), int(lm.y*h) #lm.x and lm.y store values as ratios with the total length or width. we multiply to get pixels
              self.lmlist.append([id, cx, cy])
              if draw:#showing specific landmark
                cv2.circle(img, (cx, cy),15, (255,255,0), cv2.FILLED)
        return self.lmlist

    def fingersUp(self):
        fingers = []
        if len(self.lmlist) != 0:
            if self.lmlist[self.tip_id[0]][1] < self.lmlist[self.tip_id[0] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if self.lmlist[self.tip_id[id]][2] < self.lmlist[self.tip_id[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  # webcam number zero
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
        cv2.imshow("frame", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()