import cv2
import numpy as np

def figName(contorno, width, height):
    epsilon = 0.01 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)
    print(approx)

    if len(approx) == 3:
        namefig = 'Triangulo'

    if len(approx) == 4:
        aspect_ratio = float(width) / height
        if aspect_ratio == 1:
            namefig = 'Cuadrado'
        else:
            namefig = 'Rectangulo'

    if len(approx) == 5:
        namefig = 'Pentagono'

    if len(approx) == 6:
        namefig = 'Hexagono'

    if len(approx) > 10:
        namefig = 'Circulo'

    return namefig

imagen = cv2.imread('dataset/positive/29.jpg')
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 150, 255)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
# _,cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #OpenCV 3
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4
imageHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', imageHSV)
cv2.imshow('canny', canny)
cv2.waitKey(0)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    imAux = np.zeros(imagen.shape[:2], dtype="uint8")
    imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
    maskHSV = cv2.bitwise_and(imageHSV, imageHSV, mask=imAux)
    name = figName(c, w, h)
    cv2.putText(imagen, name, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
    cv2.imshow('imagen', imagen)
    cv2.waitKey(0)