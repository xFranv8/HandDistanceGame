import cv2
from cvzone.HandTrackingModule import HandDetector


#Cam
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

#Loop
while True:
    aux, image = capture.read()
    cv2.imshow(image)
    cv2.waitKey(1)