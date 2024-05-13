import sys
import cv2
import time
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore, uic

from clasificacion_por_rostro2 import clasificar as c2
from clasificacion_por_rostro import clasificar as c1
from color import obtener_color_dominante

# Parámetros de la detección y captura
proporcion_tamano_rostro = 0.05  # Proporción del rostro sobre el área total
error_permitido = 0.20  # Error permitido para el centro
duracion_captura = 3  # Duración en segundos para la captura
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Aplicación PyQt6
class VideoCaptureApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Cargar la interfaz
        uic.loadUi("main.ui", self)

        # Obteniendo el QLabel y el botón para salir
        self.lb_camara = self.findChild(QtWidgets.QLabel, "lb_camara")
        self.btn_salir = self.findChild(QtWidgets.QPushButton, "btn_salir")
        self.btn_salir.clicked.connect(self.close_application)

        # Configurar temporizador para actualizar video
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cada 30 ms

        # Variables para captura y temporizador
        self.temporizador_captura = None
        self.imagen_capturada = None
        self.mensaje = "Esperando buen rostro..."

    def update_frame(self):
        ret, cuadro = self.cap.read()
        if not ret:
            return
        a_enviar = cuadro.copy()
        # IMPORTANTE pasar a RGB 
        cuadro = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)

        # Convertir a escala de grises y detectar rostros
        cuadro_gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = clasificador_rostros.detectMultiScale(
            cuadro_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        
        alto_frame, ancho_frame, _ = cuadro.shape
        # Centro y ejes del óvalo
        centro = (ancho_frame // 2, alto_frame // 2)
        eje_x = ancho_frame // 6
        eje_y = alto_frame // 3

        # Dibujar el óvalo en el centro del cuadro
        cv2.ellipse(
            cuadro,
            centro,
            (eje_x, eje_y),
            0,
            0,
            360,
            (255, 0, 0),
            2,
        )

        # Verificar si el rostro es una "buena captura"
        for (x, y, ancho, alto) in rostros:
            cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
            if self.es_buena_captura(x, y, ancho, alto, ancho_frame, alto_frame):
                self.mensaje = "Buen rostro detectado!"
                if self.temporizador_captura is None:
                    self.temporizador_captura = time.time()
                elif time.time() - self.temporizador_captura >= duracion_captura:
                    self.simular_flash(cuadro)
                    self.imagen_capturada = cuadro.copy()
                    break
            else:
                self.temporizador_captura = None

        # Actualizar mensaje con temporizador
        if self.temporizador_captura is not None:
            tiempo_restante = duracion_captura - (time.time() - self.temporizador_captura)
            if tiempo_restante > 0:
                self.mensaje += f" - Captura en {int(tiempo_restante)} segundos"
            else:
                self.mensaje = "Sonrie!"

        # Añadir texto con mensaje
        cv2.putText(
            cuadro,
            self.mensaje,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Convertir a formato Qt para mostrar en PyQt6
        h, w, ch = cuadro.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(
            cuadro.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_img)

        # Escalar para llenar el QLabel
        scaled_pixmap = pixmap.scaled(
            self.lb_camara.size(),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )


        self.lb_camara.setPixmap(scaled_pixmap)

        if self.imagen_capturada is not None:
            cv2.imshow("a_enviar", a_enviar)
            cv2.imwrite("./imagen.png", a_enviar)
            color = obtener_color_dominante(a_enviar)
            cv2.imshow("color", color)

            clasif1 = c1(a_enviar)
            clasif2 = c2(a_enviar)
            cv2.imshow("Clasif1", clasif1)
            cv2.imshow("Clasif2", clasif2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            self.detener_temporizador_y_mostrar_imagen()


    def es_buena_captura(self, x, y, ancho, alto, ancho_frame, alto_frame):
        area_rostro = ancho * alto
        area_frame = ancho_frame * alto_frame
        if area_rostro / area_frame >= proporcion_tamano_rostro:
            centro_frame_x = ancho_frame // 2
            centro_frame_y = alto_frame // 2
            centro_rostro_x = x + ancho // 2
            centro_rostro_y = y + ancho // 2
            if (
                abs(centro_frame_x - centro_rostro_x) <= error_permitido * ancho_frame
                and abs(centro_frame_y - centro_rostro_y) <= error_permitido * alto_frame
            ):
                return True
        return False

    def simular_flash(self, cuadro):
        cuadro_blanco = np.ones_like(cuadro) * 255
        self.lb_camara.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    cuadro_blanco.data,
                    cuadro_blanco.shape[1],
                    cuadro_blanco.shape[0],
                    cuadro_blanco.shape[1] * cuadro_blanco.shape[2],
                    QtGui.QImage.Format.Format_RGB888,
                )
            ).scaled(
                self.lb_camara.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
        time.sleep(0.25)

    def detener_temporizador_y_mostrar_imagen(self):
        self.timer.stop()
        self.cap.release()
        # Debería mostrar la imagen capturada aquí
        # Puedes abrir una nueva ventana o mostrarla en otro QLabel
        # Si quieres reiniciar, implementa la lógica para reiniciar la captura
        
    def close_application(self):
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

# Ejecutar la aplicación
app = QtWidgets.QApplication(sys.argv)
window = VideoCaptureApp()
window.show()
sys.exit(app.exec())
