import cv2
import dlib
import numpy as np
import math
from math import degrees

# Carga modelos
clasificador_cascada_rostros = cv2.CascadeClassifier("./modelos/haarcascade_frontalface_default.xml")
predictor_puntos_faciales = dlib.shape_predictor("./modelos/shape_predictor_68_face_landmarks.dat")

def classify_face_shape(image):
    imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(imagen_gris, (3, 3), 0)

    # Detección de rostros
    rostros = clasificador_cascada_rostros.detectMultiScale(
        gauss,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(rostros) == 0:
        raise ValueError("No se detectaron rostros en la imagen.")

    imagen_anotada = image.copy()

    for (x, y, w, h) in rostros:
        cv2.rectangle(imagen_anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)

        dlib_rectangulo = dlib.rectangle(x, y, x + w, y + h)
        landmarks_detectados = predictor_puntos_faciales(imagen_anotada, dlib_rectangulo).parts()
        landmarks = np.array([[p.x, p.y] for p in landmarks_detectados])

        # Cálculo de proporciones
        altura_rostro = landmarks[8, 1] - landmarks[19, 1]  # Del punto 19 al 8
        ancho_rostro = landmarks[16, 0] - landmarks[0, 0]  # De la sien izquierda a la derecha
        ancho_mandibula = landmarks[15, 0] - landmarks[1, 0]  # Ancho de la mandíbula

        # Ángulos faciales
        alpha = degrees(math.atan2(landmarks[15, 1] - landmarks[3, 1], landmarks[15, 0] - landmarks[3, 0]))

        # Clasificación del tipo de rostro
        if ancho_rostro >= altura_rostro * 0.75:
            if alpha < 160:
                shape = "Cuadrada"
            else:
                shape = "Redonda"
        elif ancho_rostro < altura_rostro * 0.75:
            if ancho_mandibula >= ancho_rostro * 0.75:
                shape = "Rectangular"
            else:
                shape = "Ovalada"

        # Anotación de la imagen
        cv2.putText(imagen_anotada, f"Tipo de rostro: {shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return imagen_anotada
