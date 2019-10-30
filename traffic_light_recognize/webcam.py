import cv2
import numpy as np
from playsound import playsound


def identifyPredominantColor(green, red, yellow):
    colors = {'green': np.count_nonzero(green), 'red': np.count_nonzero(red), 'yellow': np.count_nonzero(yellow)}
    max_value = 0
    color = ""
    for key in colors:
        if colors[key] > max_value:
            max_value = colors[key]
            color = key
    return color


def generateHsv(hsv):
    green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    red = cv2.inRange(hsv, (161, 155, 84), (179, 255, 255))
    yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    return green, red, yellow

def playAudio(color):
    print(color)
    if(color == "green"):
        playsound("green.mp3")
    elif (color == "red"):
        playsound("red.mp3")
    elif (color == "yellow"):
        playsound('yellow.mp3')


def runDetection():
    color = ""

    classifier = cv2.CascadeClassifier("cascade_semaforo_2.xml")

    cam = cv2.VideoCapture(0)

    while True:
        conectado, image = cam.read()
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = classifier.detectMultiScale(grayImage, scaleFactor=1.25, minNeighbors=7)
        for (x, y, l, a) in detections:
            cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 0), 2)
            roi_color = image[y:y + a, x:x + l]
            hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            (green, red, yellow) = generateHsv(hsv)
            colorReceived = identifyPredominantColor(green, red, yellow)
            if colorReceived != color and colorReceived != "":
                color = colorReceived
                playAudio(color)
        cv2.imshow("detect", image)
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()


runDetection()
