import cv2
import numpy as np

def figName(contorno, width, height):
    epsilon = 0.01 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)


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

    if len(approx) >= 6 or len(approx) <= 9:
        namefig = 'Hexagono'

    if len(approx) > 10:
        namefig = 'Circulo'

    else:
        print('no shape found')

    print('approx', len(approx), namefig)
    return namefig

## Read and merge
img = cv2.imread("dataset/negative/154.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blur = cv2.blur(img_hsv, ksize=(7,7))

## Gen lower mask (0-5) and upper mask (175-180) of RED
mask1 = cv2.inRange(blur, (0,50,20), (5,255,255))
mask2 = cv2.inRange(blur, (175,50,20), (180,255,255))

## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2 )
croped = cv2.bitwise_and(img, img, mask=mask)

th3 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

contour_img = np.copy(img)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 3)

## Display
# cv2.imshow("mask", mask)
# cv2.imshow("croped", croped)
cv2.imshow('contour_img', contour_img)
cv2.waitKey(0)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    imAux = np.zeros(mask.shape[:2], dtype="uint8")
    imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
    name = figName(c, w, h)
    cv2.putText(mask, name, (x, y - 5), 1, 0.8, (255, 255, 255), 1)
    cv2.imshow('imagen', mask)
    cv2.waitKey()