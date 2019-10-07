import cv2

classificador = cv2.CascadeClassifier("cascade_semaforo_2.xml")

camera = cv2.VideoCapture('esse.avi')

while camera.isOpened():
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    deteccoes = classificador.detectMultiScale(imagemCinza, scaleFactor=1.25, minNeighbors=7)
    for(x, y, l, a) in deteccoes:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
    cv2.imshow("detect", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()