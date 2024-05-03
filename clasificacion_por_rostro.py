import numpy as np
import cv2
import dlib # deteccion de puntos de referencia faciales
from sklearn.cluster import KMeans


ruta_imagen = "./rostros/rostro2.png"

# detección de rostros
modelo_deteccion_rostros = "./modelos/haarcascade_frontalface_default.xml"

# puntos de referencia faciales (predictor)
predictor_puntos_de_referencia = "./modelos/shape_predictor_68_face_landmarks.dat"

# detectar rostros en imágenes o secuencias de video utilizando un algoritmo de cascada de Haar
clasificador_cascada_rostros = cv2.CascadeClassifier(modelo_deteccion_rostros)

# Crea un Predictor de Puntos de Referencia Faciales con el Modelo
predictor_puntos_faciales = dlib.shape_predictor(predictor_puntos_de_referencia)


image = cv2.imread(ruta_imagen)

image = cv2.resize(image, (500, 500)) 

original = image.copy()

imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# suavizar imagen y reducir el ruido
gauss = cv2.GaussianBlur(imagen_gris,(3,3), 0)

# detectar rostros en imagen
"""
img pre-procesada a escala de grises
scaleFactor cuánto se reduce la imagen en cada nivel (1.05 la imagen se escala ligeramente en cada pasopara faces de diferentes tamaños)
minNeighbors Indica cuántas veces se debe detectar un objeto en un área antes de considerarlo como un rostro
minSize el tamaño mínimo del objeto a detectar. (rostros de al menos 100x100 píxeles)
flags controla el comportamiento del algoritmo de cascada. (CASCADE_SCALE_IMAGE  se aplicará el escalado a la imagen para mejorar la detección)
"""

rostros = clasificador_cascada_rostros.detectMultiScale(
    gauss,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100,100),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

print("found {0} faces!".format(len(rostros)) )

for (x,y,w,h) in rostros:
    # pintar rectangulo alrededor del rostro
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    # convierte las coordenadas del rectangulo_OpenCV a rectangulo_Dlib
    dlib_rectangulo = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    # detectar puntos de referencia faciales (landmarks)
    landmarks_detectados = predictor_puntos_faciales(image, dlib_rectangulo).parts()
    # guardar puntos de referencia faciales detectados en una matriz
    landmarks = np.matrix([[p.x,p.y] for p in landmarks_detectados])
    
    
imagen_landmarks = original.copy()

for (x,y,w,h) in rostros:
    # rectangulo
    cv2.rectangle(imagen_landmarks, (x,y), (x+w,y+h), (0,255,0), 2)
    # copia temporal
    temp = original.copy()
    # area de interes: la frente (25% de la altura de la cara)
    frente = temp[y:y+int(0.25*h), x:x+w]
    filas, columnas, canales = frente.shape
    # reorganizar para el clustering
    X = frente.reshape(filas* columnas,canales)
    """
    kmeans para la frente con 2 clusters 
    separar areas de la frente y cabello en base a el color
    """
    kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    for i in range(0,filas):
        for j in range(0,columnas):
            if y_kmeans[i*columnas+j]==True:
                frente[i][j]=[255,255,255]
            if y_kmeans[i*columnas+j]==False:
                frente[i][j]=[0,0,0]

    #Steps to get the length of frente
    #1.get midpoint of the frente
    #2.travel izquierda side and derecha side
    #the idea here is to detect the corners of frente which is the hair.
    #3.Consider the point which has change in pixel value (which is hair)

    """
    Calcular el largo de la frente
    1.obtener el punto medio del frente
    2.ir por el lado izquierdo y por el lado derecho (detectar kas esquinas)
    3.se considera el punto que tiene un cambio en el valor del pixel (que es el cabello)
    """
    frente_largo = [int(columnas/2), int(filas/2) ] # punto medio of frente
    lef=0 
    # pixel de punto central de la frente
    pixel_central_frente = frente[frente_largo[1],frente_largo[0] ]
    for i in range(0,columnas):
        # cuando hay un cambio del color
        if frente[frente_largo[1],frente_largo[0]-i].all()!=pixel_central_frente.all():
            lef=frente_largo[0]-i
            break
    izquierda = [lef,frente_largo[1]]
    rig=0
    for i in range(0,columnas):
        # cuando hay un cambio del color
        if frente[frente_largo[1],frente_largo[0]+i].all()!=pixel_central_frente.all():
            rig = frente_largo[0]+i
            break
    derecha = [rig,frente_largo[1]]
    
# pintar linea1 (frente)

#  diferencia de valores -> distancia
linea1 = np.subtract(derecha+y,izquierda+x)[0]
# pintar linea en la imagen
cv2.line(imagen_landmarks, tuple(x+izquierda), tuple(y+derecha), color=(0,255,0), thickness = 2)
# colocar nombre linea en la imagen
cv2.putText(imagen_landmarks,' linea 1',tuple(x+izquierda),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
# pintar puntos de referencia
cv2.circle(imagen_landmarks, tuple(x+izquierda), 5, color=(255,0,0), thickness=-1)
cv2.circle(imagen_landmarks, tuple(y+derecha), 5, color=(255,0,0), thickness=-1)

# pintar linea en la imagen
punto_linea_izquierda = (landmarks[1,0],landmarks[1,1])
punto_linea_derecha = (landmarks[15,0],landmarks[15,1])
linea2 = np.subtract(punto_linea_derecha,punto_linea_izquierda)[0]
cv2.line(imagen_landmarks, punto_linea_izquierda,punto_linea_derecha,color=(0,255,0), thickness = 2)
cv2.putText(imagen_landmarks,' linea 2',punto_linea_izquierda,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
cv2.circle(imagen_landmarks, punto_linea_izquierda, 5, color=(255,0,0), thickness=-1)    
cv2.circle(imagen_landmarks, punto_linea_derecha, 5, color=(255,0,0), thickness=-1)    


punto_linea_izquierda = (landmarks[3,0],landmarks[3,1])
punto_linea_derecha = (landmarks[13,0],landmarks[13,1])
linea3 = np.subtract(punto_linea_derecha,punto_linea_izquierda)[0]
cv2.line(imagen_landmarks, punto_linea_izquierda,punto_linea_derecha,color=(0,255,0), thickness = 2)
cv2.putText(imagen_landmarks,' linea 3',punto_linea_izquierda,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
cv2.circle(imagen_landmarks, punto_linea_izquierda, 5, color=(255,0,0), thickness=-1)    
cv2.circle(imagen_landmarks, punto_linea_derecha, 5, color=(255,0,0), thickness=-1)    


punto_linea_abajo = (landmarks[8,0],landmarks[8,1])
punto_linea_arriba = (landmarks[8,0],y)
linea4 = np.subtract(punto_linea_abajo,punto_linea_arriba)[1]
cv2.line(imagen_landmarks,punto_linea_arriba,punto_linea_abajo,color=(0,255,0), thickness = 2)
cv2.putText(imagen_landmarks,' linea 4',punto_linea_abajo,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
cv2.circle(imagen_landmarks, punto_linea_arriba, 5, color=(255,0,0), thickness=-1)    
cv2.circle(imagen_landmarks, punto_linea_abajo, 5, color=(255,0,0), thickness=-1)    

# dispersion de las medidas del rostro (anchos)
similitud = np.std([linea1,linea2,linea3])
# cruz del rostro
ovalsimilitud = np.std([linea2,linea4])

# arcotangente para obtener angulos
# puntos de la mandibula
ax,ay = landmarks[3,0],landmarks[3,1]
bx,by = landmarks[4,0],landmarks[4,1]
cx,cy = landmarks[5,0],landmarks[5,1]
dx,dy = landmarks[6,0],landmarks[6,1]

import math
from math import degrees

alpha0 = math.atan2(cy-ay,cx-ax)
alpha1 = math.atan2(dy-by,dx-bx)
alpha = alpha1-alpha0
angulo = abs(degrees(alpha))
angulo = 180-angulo

for i in range(1):
  if similitud < 10:
    if angulo < 160:
      print('forma cuadrada. Las líneas de la mandíbula son más angulares')
      break
    else:
      print('forma redonda. Las líneas de la mandíbula no son tan angulares')
      break
  if linea3 > linea1:
    if angulo < 160:
      print('forma de triángulo. La frente es más ancha')
      break
  if ovalsimilitud < 10:
    print('forma de diamante. Las líneas 2 y 4 son similares, y la línea 2 es ligeramente más grande')
    break
  if linea4 > linea2:
    if angulo < 160:
      print('forma rectangular. La longitud del rostro es la más grande y las líneas de la mandíbula son angulares')
      break
    else:
      print('forma alargada. La longitud del rostro es la más grande y las líneas de la mandíbula no son angulares')
      break
  print("Error")

output = np.concatenate((original, imagen_landmarks), axis=1)
cv2.imshow('salida', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
