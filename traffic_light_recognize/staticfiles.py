import cv2

classificador = cv2.CascadeClassifier("cascade_semaforo_2.xml")

imagem = cv2.imread("ImagesTest/10.jpg")

imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

deteccoes= classificador.detectMultiScale(imagemcinza, scaleFactor=1.25, minNeighbors=7)

for(x, y, l, a) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 255, 0), 2)

cv2.imshow("", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

