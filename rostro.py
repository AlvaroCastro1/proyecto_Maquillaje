import sys
import cv2
import time
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore, uic
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt
import os

# Aplicación PyQt6
class VideoCaptureApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Cargar la interfaz
        uic.loadUi("rostro.ui", self)

        # Configurar botones
        self.btn_salir.clicked.connect(self.close_application)


        # Configurar temporizador para actualizar video
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cada 30 ms

        # Variables para captura y temporizador
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')

        # Lista de máscaras  con canal alfa
        self.masks = [
            cv2.imread('mascaras/redondo.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('mascaras/ovalado.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('mascaras/alargado.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('mascaras/cuadrado.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('mascaras/corazon.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('mascaras/triangulo.png', cv2.IMREAD_UNCHANGED)
        ]

        # Lista de los nombres
        self.labels = ["Rostro Redondo", "Rostro Ovalado", "Rostro Alargado", "Rostro Cuadrado", "Rostro Corazon", "Rostro Triangulo"]
        self.label_index = 0
        self.cuadro = None
        self.capture_directory = "capturas"
        if not os.path.exists(self.capture_directory):
            os.makedirs(self.capture_directory)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_R:
            self.continuar_clasifiacion=True
            self.temporizador_captura = None
        if event.key() == Qt.Key.Key_L:
            self.label_index = (self.label_index + 1) % len(self.labels)
        if event.key() == Qt.Key.Key_Q:
            self.close_application()
        if event.key() == Qt.Key.Key_C:
            capture_filename = os.path.join(self.capture_directory, 'capture_{}.png'.format(cv2.getTickCount()))
            self.cuadro = cv2.cvtColor(self.cuadro, cv2.COLOR_BGR2RGB)
            cv2.imwrite(capture_filename, self.cuadro)
            print(f'Captura guardada como {capture_filename}')
        else:
            super().keyPressEvent(event)

    def update_frame(self):
        ret, cuadro = self.cap.read()

        cuadro = cv2.flip(cuadro, 1)
        cuadro = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Redimensionar la máscara actual para que coincida con el tamaño de la cara detectada
            mask_resized = cv2.resize(self.masks[self.label_index], (w, h), interpolation=cv2.INTER_AREA)
            
            # Obtener las coordenadas de la esquina superior izquierda de la región de la cara
            x_start = x
            y_start = y
            
            # Obtener las coordenadas de la esquina inferior derecha de la región de la cara
            x_end = x + w
            y_end = y + h
            
            mask_resized = mask_resized[:min(h, cuadro.shape[0]-y_start), :min(w, cuadro.shape[1]-x_start)]
            
            # Colocar la máscara en la cara
            alpha_mask = mask_resized[:, :, 3] / 255.0
            alpha_cuadro = 1.0 - alpha_mask

            for c in range(0, 3):
                cuadro[y_start:y_end, x_start:x_end, c] = (alpha_mask * mask_resized[:, :, c] + alpha_cuadro * cuadro[y_start:y_end, x_start:x_end, c])

            # Dibujar un rectángulo alrededor de la cara detectada
            cv2.rectangle(cuadro, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Agregar la etiqueta al lado de la cara corrsponde a la imagen 
            label = self.labels[self.label_index]
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_x = x + w + 10
            label_y = y + int(h / 2)
            
            cv2.putText(cuadro, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Convertir a formato Qt para mostrar en PyQt6
        h, w, ch = cuadro.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(
            cuadro.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )
        self.cuadro = cuadro.copy()
        pixmap = QtGui.QPixmap.fromImage(q_img)

        # Escalar para llenar el QLabel
        scaled_pixmap = pixmap.scaled(
            self.lb_camara.size(),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.lb_camara.setPixmap(scaled_pixmap)
  
    def close_application(self):
        # from menu_inicial import MainApp
        # self.new_window = MainApp()
        # self.new_window.show()
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
