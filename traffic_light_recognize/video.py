import cv2
import numpy as np

classificador = cv2.CascadeClassifier("cascade_semaforo_2.xml")

camera = cv2.VideoCapture('esse.avi')

while camera.isOpened():
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    deteccoes = classificador.detectMultiScale(imagemCinza, scaleFactor=1.25, minNeighbors=7)
    for(x, y, l, a) in deteccoes:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 0), 2)
        roi_color = imagem[y:y + a, x:x + l]
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
        red = cv2.inRange(hsv, (161, 155, 84), (179, 255, 255))
        yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        print('green: ', np.count_nonzero(green))
        print('yellow: ', np.count_nonzero(yellow))
        print('red: ', np.count_nonzero(red))
    cv2.imshow("detect", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()