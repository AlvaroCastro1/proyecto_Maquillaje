import sys
from PyQt6 import QtWidgets, uic


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Cargar la interfaz
        uic.loadUi("menu_inicial.ui", self)

        self.btn_color_forma = self.findChild(QtWidgets.QPushButton, "btn_color_forma")
        self.btn_filtro = self.findChild(QtWidgets.QPushButton, "btn_filtro")
        self.btn_maquillaje = self.findChild(QtWidgets.QPushButton, "btn_maquillaje")


        self.btn_color_forma.clicked.connect(self.color_forma)

    def close_application(self):
        self.close()

    def closeEvent(self, event):
        event.accept()

    def color_forma(self):
        from color_y_forma import VideoCaptureApp
        self.new_window = VideoCaptureApp()
        self.new_window.show()
        self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
