import cv2
import numpy as np
from math import isclose
from pynput.keyboard import Key, Controller


cap = cv2.VideoCapture(0)
tracker = []
keyboard = Controller()

didMove = []


while True:
    keyboard.release(Key.right)
    keyboard.release(Key.left)
    
    
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    #cv2.imshow('mask before morphologyEx', mask)
    kernel = np.ones((5,5),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    (_,cnts,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        x,y,w,h = (0,)*4
        current_biggest = 0
        rectToDraw = None
        for i in range(len(cnts)):
            epsilon = 0.01*cv2.arcLength(cnts[i],True)
            approx = cv2.approxPolyDP(cnts[i],epsilon,True)
            rect = cv2.minAreaRect(approx)
            
            (x, y), (width, height), angle = rect
            if width*height > current_biggest:
                current_biggest = width*height
                rectToDraw = rect
            
        if rectToDraw:
            box = cv2.boxPoints(rectToDraw)
            box = np.int0(box)
            cv2.drawContours(res,[box],0,(0,0,255),2)
            (x, y), (width, height), angle = rectToDraw
            tracker.append({"x": x, "y": y, "area": current_biggest})

        if len(tracker) > 1 and isclose(tracker[-1]["area"], tracker[-2]["area"], rel_tol=0.9):
            if tracker[-1]["y"] > tracker[-2]["y"]+50:
                if shouldMove():    
                    print("down")
                    keyboard.press(Key.down)
            
    def shouldMove(max = 2):
        moveOrNot = True
        upTo = max if len(didMove) > max else 1 if len(didMove) is 0 else len(didMove)
        for i in range(1, upTo):
            if didMove[-i] is True:
                didMove.append(False)
                return False
        didMove.append(True)
        return True
    
    
    cv2.imshow('original image', frame)
    #cv2.imshow('mask after morphologyEx', mask)
    cv2.imshow('after bitwise_and with contours', res)

    
    k=cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()


#sys.stdout.write("\r" + ("_"*(square["x"]-len(self.body)))+("O"*len(self.body))+("_"*(width-square["x"])))
#sys.stdout.flush()
