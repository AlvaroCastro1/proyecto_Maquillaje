import cv2
import dlib
import numpy as np
from collections import Counter

# Función para detectar el color dominante de un rostro en una imagen
def obtener_color_dominante(imagen):
    # Convertir la imagen a RGB (OpenCV usa BGR por defecto)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Inicializar el detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()

    # Detectar rostros
    rostros = detector(imagen_rgb)

    # Verificar si se encontraron rostros
    if not rostros:
        print("No se encontraron rostros en la imagen.")
        return None
    
    # Lista para almacenar los colores de la piel en los rostros detectados
    colores_piel = []

    # Iterar sobre los rostros detectados para obtener sus colores
    for rostro in rostros:
        x, y, w, h = rostro.left(), rostro.top(), rostro.width(), rostro.height()
        region_rostro = imagen_rgb[y:y + h, x:x + w]
        
        # Aplanar la región y añadir los colores
        colores = region_rostro.reshape(-1, region_rostro.shape[-1])
        colores_piel.extend(colores)

    # Calcular el color dominante entre los colores obtenidos
    contador_colores = Counter(map(tuple, colores_piel))
    color_dominante = contador_colores.most_common(1)[0][0]  # RGB más frecuente

    # Convertir el color dominante a BGR para usar con OpenCV
    color_dominante_bgr = tuple(reversed(color_dominante))

    # Crear una imagen de 100x100 con el color dominante
    imagen_color = np.zeros((100, 100, 3), dtype=np.uint8)
    imagen_color[:, :] = color_dominante_bgr  # Rellenar con el color dominante
    
    return imagen_color

# Ejemplo de uso de la función
if __name__ == "__main__":
    # Cargar la imagen
    image_path = "./rostros/rostro.jpg"
    imagen = cv2.imread(image_path)

    # Obtener la imagen con el color dominante
    imagen_color = obtener_color_dominante(imagen)

    cv2.imshow("Color Dominante", imagen_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
