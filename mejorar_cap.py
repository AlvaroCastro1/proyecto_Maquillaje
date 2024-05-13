import cv2
import numpy as np

def mejorar_calidad(frame):
    """
    Mejora la calidad del frame mediante una combinación de técnicas de preprocesamiento.
    
    :param frame: El frame de entrada a mejorar.
    :return: El frame mejorado.
    """
    
    # Reducción de ruido usando suavizado gaussiano
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Ajuste de contraste y brillo
    alpha = 1.3  # Contraste
    beta = 20    # Brillo
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Corrección de gamma para realzar colores
    gamma = 1.2  # Ajusta según sea necesario
    invGamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame, gamma_table)
    
    # Suavizado mediano para eliminar ruido adicional
    frame = cv2.medianBlur(frame, 5)
    
    # Opción para resaltar detalles mediante detección de bordes (opcional, puede comentarse si no es necesario)
    # edges = cv2.Canny(frame, 100, 200)
    
    # Redimensionar para aumentar resolución (opcional)
    scale_factor = 1.5  # Escala el tamaño por 1.5
    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    return frame

import pywt  # Librería para Wavelets

def mejorar_calidad_alternativa(frame):
    """
    Mejora la calidad del frame usando un enfoque alternativo.
    
    :param frame: El frame de entrada a mejorar.
    :return: El frame mejorado.
    """
    
    # Conversión a escala de grises para algunas técnicas de procesamiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Normalización del histograma para mejorar el contraste
    gray_frame = cv2.equalizeHist(gray_frame)
    
    # Convertir de nuevo a color después de la normalización
    frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Reducción de ruido usando la Transformada Wavelet
    coeffs2 = pywt.dwt2(frame, 'db1')  # Usando wavelet 'db1'
    LL, (LH, HL, HH) = coeffs2
    # Reducir componentes de alta frecuencia para eliminar ruido
    LL = cv2.GaussianBlur(LL, (5, 5), 0)
    # Reconstruir la imagen después de reducción de ruido
    frame = pywt.idwt2((LL, (LH, HL, HH)), 'db1')
    
    # Realce de bordes para aumentar detalles
    frame = cv2.addWeighted(frame, 1.5, cv2.GaussianBlur(frame, (9, 9), 0), -0.5, 0)
    
    return frame

def clahe_contrast(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica CLAHE para mejora de contraste.
    
    :param frame: El frame de entrada.
    :param clip_limit: Umbral para contraste.
    :param tile_grid_size: Tamaño de la rejilla para el cálculo local del histograma.
    :return: Frame con mejora de contraste.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_frame = clahe.apply(gray_frame)
    frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return frame


if "__main__" == __name__:
    original = cv2.imread("./imagen.png")

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original", original)

    salida=mejorar_calidad(original)
    cv2.namedWindow("salida", cv2.WINDOW_NORMAL)
    cv2.imshow("salida", salida)

    salida1=mejorar_calidad_alternativa(original)
    cv2.namedWindow("salida1", cv2.WINDOW_NORMAL)
    cv2.imshow("salida1", salida)

    salida2=clahe_contrast(original)
    cv2.namedWindow("salida2", cv2.WINDOW_NORMAL)
    cv2.imshow("salida2", salida)

    cv2.waitKey()
    cv2.destroyAllWindows()