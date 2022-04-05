import random
import cv2, math, time
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Cam
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = []
x = np.reshape(x, (-1,1))
for i in range(20, 105, 5):
    y.append(i)


poly_regs = PolynomialFeatures(degree=2)
x_poly = poly_regs.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

cx, cy = 200, 200
color = (255, 0, 255)
count = 0
points = 0
pressed = False
unpressed = True

initial_time = time.time()

while True:
    aux, image = capture.read()
    hands = detector.findHands(image)

    if time.time() - initial_time < 30:
        if hands:
            hand_points = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']

            x1 = hand_points[5][0]
            y1 = hand_points[5][1]
            x2 = hand_points[17][0]
            y2 = hand_points[17][1]

            distance = math.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2)
            distance_poly = poly_regs.fit_transform([[distance]])
            cm = lin_reg.predict(distance_poly)

            print(distance, " : ", cm)

            if cm < 35 and (x < cx < x + w) and (y < cy < y + h):
                pressed = True
            else:
                unpressed = True

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cvzone.putTextRect(image, f'{int(cm)} cm', (x+5, y-10))


        cv2.circle(image, (cx, cy), 30, color, cv2.FILLED)
        cv2.circle(image, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
        cv2.circle(image, (cx, cy), 20, (0, 0, 0), 2)
        cv2.circle(image, (cx, cy), 30, (255, 255, 255), 3)


        if pressed and unpressed:
            count += 1
            color = (0, 255, 0)
            unpressed = False
            pressed = False
            if count == 3:
                cx = random.randint(100, 1100)
                cy = random.randint(100, 600)
                color = (255, 0, 255)
                count = 0
                points += 1

        cvzone.putTextRect(image, f'Time: {30 - (time.time() - initial_time)}', (75, 1000), scale=3, offset=25)
        cvzone.putTextRect(image, f'Points: {points}', (1000, 75), scale=3, offset=25)
    else:
        pass
    cv2.imshow("Hand Detector", image)
    cv2.waitKey(1)




















