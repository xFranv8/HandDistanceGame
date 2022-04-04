import cv2, math
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
for i in range(20, 105, 5):
    y.append(i)

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

while True:
    aux, image = capture.read()
    hands, image = detector.findHands(image)

    if hands:
        hand_points = hands[0]['lmList']
        x1, y1 = hand_points[5]
        x2, y2 = hand_points[17]
        distance = math.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2)
        cm = lin_reg.predict(distance)
        print(distance, " : ", cm)
    cv2.imshow("Hand Detector", image)
    cv2.waitKey(1)