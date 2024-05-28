import sys
import cv2
import time
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore, uic
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt

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
        self.lb_color_piel = self.findChild(QtWidgets.QLabel, "lb_color_piel")
        self.lb_forma = self.findChild(QtWidgets.QLabel, "lb_forma")
        self.btn_salir = self.findChild(QtWidgets.QPushButton, "btn_salir")
        self.btn_forma_aplicar = self.findChild(QtWidgets.QPushButton, "btn_forma_aplicar")
        
        # Configurar botones
        self.btn_salir.clicked.connect(self.close_application)
        #self.btn_forma_aplicar.clicked.connect(self.aqui_va_la_accion)

        # Configurar temporizador para actualizar video
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cada 30 ms

        # Variables para captura y temporizador
        self.temporizador_captura = None
        self.imagen_capturada = None
        self.mensaje = "Esperando buen rostro..."
        self.continuar_clasifiacion=True

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_R:
            self.continuar_clasifiacion=True
            self.temporizador_captura = None
        else:
            super().keyPressEvent(event)


    def update_frame(self):
        ret, cuadro = self.cap.read()
        if not ret:
            return
        # IMPORTANTE pasar a RGB 
        cuadro = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
        cuadro = cv2.flip(cuadro, 1)
        cuadro_capturado = cuadro.copy()
        cuadro_capturado = cv2.cvtColor(cuadro_capturado, cv2.COLOR_BGR2RGB)        

        if self.continuar_clasifiacion:
            # print("haciendo clasificacion")
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
            cv2.ellipse(cuadro,centro,(eje_x, eje_y),0,0,360,(255, 0, 0),2)

            # Verificar si el rostro es una "buena captura"
            for (x, y, ancho, alto) in rostros:
                cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
                if self.es_buena_captura(x, y, ancho, alto, ancho_frame, alto_frame):
                    self.mensaje = "Buen rostro detectado!"
                    if self.temporizador_captura is None:
                        self.temporizador_captura = time.time()
                    elif time.time() - self.temporizador_captura >= duracion_captura:
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
                    self.continuar_clasifiacion = False
                    
                    # obtener color y mostrarlo en una de las etiquetas
                    color = obtener_color_dominante(cuadro_capturado)
                    color1 = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    altura, ancho, canales = color1.shape
                    bytes_per_line = 3 * ancho
                    q_image = QImage(color1.data, ancho, altura, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.lb_color_piel.setPixmap(pixmap)

                    # obtener forma y mostrarlo en una de las etiquetas
                    clasif1,forma1 = c1(cuadro_capturado)
                    clasif2,forma2 = c2(cuadro_capturado)
                    self.lb_forma.setText(f"formas: \n{forma1}\n{forma2}")
                    # cv2.imshow("c", clasif1)
                    # cv2.imshow("color", color)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    # Configurar fuente y tamaño
                    fuente = QFont('Arial', 12)
                    self.lb_forma.setFont(fuente)

            # Añadir texto con mensaje
            cv2.putText(cuadro,self.mensaje,(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)

        else:
            # print("ya se acabo la clasificacion")
            pass
        
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
        
    def close_application(self):
        from menu_inicial import MainApp
        self.new_window = MainApp()
        self.new_window.show()
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    # Ejecutar la aplicación
    app = QtWidgets.QApplication(sys.argv)
    window = VideoCaptureApp()
    window.show()
    sys.exit(app.exec())
