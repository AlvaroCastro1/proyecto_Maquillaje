import cv2
import time
import datetime
import numpy as np

# porcentaje del tamanio (10%, 20,%)
tamanio_rostro = 0.05
# Cronómetro para la captura
capture_timer = None
capture_duration = 3  # Duración de la cuenta regresiva en segundos
captured_frame = None

# Clasificador para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# garantizar buena calidad de la imagen inicial
def is_buena_imagen(x, y, w, h, ancho_frame, frame_ancho):
    area_del_rostro = w * h
    frame_area = ancho_frame * frame_ancho
    
    # Determina si el rostro es suficientemente grande y centrado
    if area_del_rostro / frame_area >= tamanio_rostro:
        frame_centro_x = ancho_frame // 2
        frame_centro_y = frame_ancho // 2
        face_centro_x = x + w // 2
        face_centro_y = y + h // 2
        
        if abs(frame_centro_x - face_centro_x) <= 0.2 * ancho_frame and abs(frame_centro_y - face_centro_y) <= 0.2 * ancho_frame:
            return True
    
    return False

# Captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Bucle de captura de imagen
while True:
    ret, frame = cap.read()
    
    frame_height, frame_width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversión a escala de grises
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    message = "Esperando buen rostro..."  # Mensaje predeterminado
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja el rectángulo
        
        if is_buena_imagen(x, y, w, h, frame_width, frame_height):
            message = "Buen rostro detectado!"
            if capture_timer is None:
                # Inicia el cronómetro cuando se detecta un buen rostro
                capture_timer = time.time()
            elif time.time() - capture_timer >= capture_duration:
                captured_frame = frame.copy()  # Captura la imagen
                break  # Sale del bucle interno
        else:
            # Reinicia el cronómetro si el rostro ya no es bueno
            capture_timer = None

    # Mostrar el tiempo restante en la cuenta regresiva
    if capture_timer is not None:
        time_remaining = capture_duration - (time.time() - capture_timer)
        if time_remaining > 0:
            message += f" - Captura en {int(time_remaining)} segundos"
        else:
            message += " - ¡Captura ahora!"

    # Mostrar el mensaje y el video en tiempo real
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Deteccion de Rostros', frame)

    # Si se ha capturado una imagen, cerrar la ventana principal y mostrar la imagen en una nueva ventana
    if captured_frame is not None:
        # Cerrar la ventana principal
        cap.release()
        cv2.destroyAllWindows()

        # Mostrar la imagen capturada en una nueva ventana
        cv2.imshow("Imagen Capturada", captured_frame)

        # Permitir que el usuario decida si quiere volver a capturar
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):  # Volver a capturar
                # Reiniciar el cronómetro y el bucle
                capture_timer = None
                captured_frame = None  # Eliminar la captura anterior
                cap = cv2.VideoCapture(0)  # Reabrir la webcam
                break
            elif key == ord('q'):  # Salir
                cv2.destroyAllWindows()
                exit()

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
