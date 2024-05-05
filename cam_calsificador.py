import cv2
import time
import numpy as np

# from clasificador.clasificador import clasificar_rostro

# porcentaje del tamaño del rostro (10%, 20%, ...)
proporcion_tamano_rostro = 0.05

# error permitido para el centro de captura (del total del video)
error_permitido = 0.20

# cuenta regresiva en segundos
duracion_captura = 3
# temporizador para la captura
temporizador_captura = None
imagen_capturada = None

# Cargar el clasificador de Haar para detección de rostros
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def simular_flash():
    # simular el flash antes de la captura en la misma ventana
    cuadro_blanco = np.ones_like(cuadro) * 255
    cv2.imshow("flash", cuadro_blanco)
    cv2.waitKey(250)  # mantener por 250ms

def es_buena_captura(x, y, ancho, alto, ancho_frame, alto_frame):
    # area rostro
    area_rostro = ancho * alto
    #area total    
    area_frame = ancho_frame * alto_frame
    
    # validacion del tamaño del rostro
    if area_rostro / area_frame >= proporcion_tamano_rostro:
        # centro del frame
        centro_frame_x = ancho_frame // 2
        centro_frame_y = alto_frame // 2
        
        # centro del rostro
        centro_rostro_x = x + ancho // 2
        centro_rostro_y = y + alto // 2
        
        # si el centro del rostro esta CERCA del centro del cuadro
        if abs(centro_frame_x - centro_rostro_x) <= error_permitido * ancho_frame and abs(centro_frame_y - centro_rostro_y) <= error_permitido * alto_frame:
            """
            la captura del rostro es "buena" considerenado:
                es de buen tamaño
                que esta centrado 
            """
            return True
    # rostro no cumple con los requisitos
    return False

# Captura de video desde la webcam
camara = cv2.VideoCapture(0)

while True:
    # Leer un frame a frame de la camara
    ret, cuadro = camara.read()
    if not ret:
        break  # Salir del bucle si no se puede leer
    
    alto_frame, ancho_frame, _ = cuadro.shape
    cuadro_gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    
    # mantener cuadro original para despues guardar
    cuadro_original = cuadro.copy()
    # Detectar rostros en el cuadro
    rostros = clasificador_rostros.detectMultiScale(cuadro_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    mensaje = "Esperando buen rostro..."
    
    # Dibujar el ovalo de referencia para que se coloque el rostro
    centro = (ancho_frame // 2, alto_frame // 2)
    # Tamaño del óvalo
    ejes = (ancho_frame // 5, alto_frame // 3)
    # Ángulo del óvalo
    angulo = 0
    inicio_angulo = 0
    fin_angulo = 360
    color = (255, 0, 0)
    grosor = 2
    #cv2.ellipse(imagen2colocar, centro, tuplaEjer, angulo, inicio_angulo, fin_angulo, color, grosor)
    cv2.ellipse(cuadro, centro, ejes, angulo, inicio_angulo, fin_angulo, color, grosor)
    
    for (x, y, ancho, alto) in rostros:
        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        
        # Verificar si el rostro es "bueno"
        if es_buena_captura(x, y, ancho, alto, ancho_frame, alto_frame):
            mensaje = "Buen rostro detectado!"
            if temporizador_captura is None:
                # comienza el temporizador si se es un buen rostro
                temporizador_captura = time.time()
            elif time.time() - temporizador_captura >= duracion_captura:
                # Si el temporizador ha expirado, guardar la imagen y salir de la captura de video
                simular_flash()
                imagen_capturada = cuadro.copy()
                break 
        else:
            # Si la captura cambia y/o no es bueno, reiniciar el temporizador
            temporizador_captura = None

    # mostrar el tiempo restante de la cuenta regresiva, solo si esta en progreso
    if temporizador_captura is not None:
        tiempo_restante = duracion_captura - (time.time() - temporizador_captura)
        if tiempo_restante > 0:
            mensaje += f" - Captura en {int(tiempo_restante)} segundos"
        else:
            mensaje += " - ¡Sonrie!"
            cuadro = np.ones_like(cuadro)*255


    # Mostrar el mensaje y el video en tiempo real
    cv2.putText(cuadro, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Deteccion de Rostros', cuadro)

    # Si se ha capturado una imagen, detener la captura y mostrar la imagen capturada
    if imagen_capturada is not None:
        # Detener la captura de video
        camara.release()
        cv2.destroyAllWindows()
        
        # Mostrar la imagen capturada en una nueva ventana
        cv2.imshow("Imagen Capturada", cuadro_original)
        # rostro_clasificado = clasificar_rostro(cuadro_original)
        # cv2.imshow("Figura", rostro_clasificado)
        
        # Preguntar al usuario si quiere volver a capturar
        while True:
            tecla = cv2.waitKey(0) & 0xFF
            if tecla == ord('r'):  # Reintentar captura
                # Reiniciar el temporizador y el bucle
                temporizador_captura = None
                imagen_capturada = None  # Limpiar la imagen capturada
                camara = cv2.VideoCapture(0)  # Reabrir la webcam
                break
            elif tecla == ord('q'):  # Salir
                cv2.destroyAllWindows()
                exit()  # Terminar el programa
    
    # Si se presiona 'q', salir del bucle principal
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
camara.release()
cv2.destroyAllWindows()
