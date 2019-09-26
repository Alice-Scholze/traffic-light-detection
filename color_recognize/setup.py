import cv2
import numpy as np

img = cv2.imread('nored.png')
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0,20,20), (8,255,255))
mask2 = cv2.inRange(hsv, (170,20,20), (180,255,255))
mask = cv2.bitwise_or(mask1, mask2)
dst = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("dst", dst)
cv2.imwrite("__.png", dst)
cv2.waitKey()