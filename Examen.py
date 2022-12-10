import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline

def show(img, titulo):
    plt.figure(figsize=(7,7))
    plt.title(titulo)
    plt.imshow(img)
    plt.show()

image = cv2.imread('./jitomaton.jpg')
#convertir de formato BGR a RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = image
show(original, "Imagen original " + str(image.shape))

# Aplicar filtro suavizado
image = cv2.blur(image, (150, 150),0)
#Convertir imágen a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# separar canales hsv
h, s, v = cv2.split(hsv)
# aplicar binarización OTSU
_, thr = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmentada = cv2.bitwise_and(original, original, mask = thr)

contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Imagen para colocar contornos
minRectImage = segmentada.copy()

croppedImg = 0 

def encontrar_recta(dot_1, dot_2):
  m1 = [dot_1[0], dot_1[1], 1]
  m2 = [dot_2[0], dot_2[1], 1]

  xs = [m1[0], m2[0]]
  ys = [m1[1], m2[1]]

  recta = np.cross(m1, m2)

  a = recta[0]
  b = recta[1]
  c = recta[2]

  return a, b, c 

def calcular_distancia(dot_1, dot_2):
  return math.sqrt( (dot_1[0] - dot_2[0])**2 + (dot_1[1] - dot_2[1])**2 )

def trazar_recta(imagen, rectas):
  
  rectas.sort(key = lambda x: x[4])

  for i in range(len(rectas) - 2, len(rectas)):

    cv.line(original, (rectas[i][0], rectas[i][1]), (rectas[i][2], rectas[i][3]), (255,0,255), 8)

    print('Puntos de medicion')
    print('Punto: x = ', rectas[i][0], ' y = ', rectas[i][1])
    print('Punto: x = ', rectas[i][2], ' y = ', rectas[i][3])
    print('Distancia: ', rectas[i][4])

  return imagen

rectas = []

for i, c in enumerate(contours):

  # Obtener area
  contourArea = cv2.contourArea(c)
  # Minima area de contorno
  minArea = 200000

  # Buscar contornos grandes
  if contourArea > minArea:

    boundingRectangle = cv2.minAreaRect(c)
    # Puntos del rectangulo
    rectanglePoints = cv2.boxPoints(boundingRectangle)

    a1, b1, c1 = encontrar_recta(rectanglePoints[0], rectanglePoints[3])
    xs_1 = (int) ( (rectanglePoints[0][0] + rectanglePoints[3][0]) /2 )
    ys_1 = (int) (-(a1*xs_1 + c1) / b1)

    a2, b2, c2 = encontrar_recta(rectanglePoints[1], rectanglePoints[2])
    xs_2 = (int) ( (rectanglePoints[1][0] + rectanglePoints[2][0]) /2 )
    ys_2 = (int) (-(a1*xs_2 + c2) / b2)

    dist = calcular_distancia((xs_1, ys_1), (ys_2, ys_2))

    rectas.append((xs_1, ys_1, xs_2, ys_2, dist))

    # cv.line(original, (xs_1, ys_1), (xs_2, ys_2), (255,0,255), 8)

    # Convert float array to int array:
    rectanglePoints = np.intp(rectanglePoints)
    # Draw the min area rectangle:
    cv2.drawContours(minRectImage, [rectanglePoints], 0, (255, 255, 255), 4)

show(minRectImage, "Regiones de interes")

original = trazar_recta(original, rectas)
show(original, "Rectas")
