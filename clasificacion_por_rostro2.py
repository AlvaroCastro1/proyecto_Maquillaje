import numpy as np
import cv2
import dlib

def calcular_distancias(puntos_referencia):
    # Calcular distancias clave entre puntos de referencia
    altura_vertical = np.sqrt((puntos_referencia[15].x - puntos_referencia[1].x) ** 2 + 
                              (puntos_referencia[15].y - puntos_referencia[1].y) ** 2)
    
    ancho_frente = np.sqrt((puntos_referencia[77].x - puntos_referencia[78].x) ** 2 + 
                           (puntos_referencia[77].y - puntos_referencia[78].y) ** 2)
    
    ancho_pomulos = np.sqrt((puntos_referencia[3].x - puntos_referencia[8].x) ** 2 + 
                            (puntos_referencia[3].y - puntos_referencia[8].y) ** 2)
    
    ancho_mandibula = np.sqrt((puntos_referencia[71].x - puntos_referencia[78].x) ** 2 + 
                              (puntos_referencia[71].y - puntos_referencia[78].y) ** 2)
    
    return altura_vertical, ancho_frente, ancho_pomulos, ancho_mandibula

def clasificar_forma_rostro(distancias):
    altura_vertical, ancho_frente, ancho_pomulos, ancho_mandibula = distancias
    
    
    relacion_vertical_frente = altura_vertical / ancho_frente

    relacion_pomulos_mandibula = ancho_pomulos / ancho_mandibula
    
    """
    Ovalado: Se caracteriza por una relación de altura vertical/ancho de la frente mayor que 1.3 y una relación de ancho de pómulos/ancho de la mandíbula mayor que 1.0.
    Alargado (Oblong): Cuando la relación de altura vertical/ancho de la frente es bastante alta (por ejemplo, mayor que 1.5) y la relación de ancho de pómulos/ancho de la mandíbula es baja (por ejemplo, menos de 0.8).
    Redondo: Una relación de altura vertical/ancho de la frente menor que 1.2 y una relación de ancho de pómulos/ancho de la mandíbula bastante alta (por ejemplo, mayor que 1.2).
    Cuadrado: Cuando el ancho de la frente es mayor que el ancho de los pómulos y la mandíbula.
    Corazón: Si el ancho de los pómulos es mayor que el ancho de la frente y la mandíbula.
    Diamante: Cuando la mandíbula es más ancha que el resto.
    Triangular: Si ninguna de las condiciones anteriores se cumple, se podría considerar un rostro triangular.
    """

    if relacion_vertical_frente >= 1.3 and relacion_pomulos_mandibula >= 1.0:
        forma_rostro = "Ovalado"
    elif relacion_vertical_frente >= 1.5 and relacion_pomulos_mandibula < 0.8:
        forma_rostro = "Alargado"
    elif relacion_vertical_frente < 1.2 and relacion_pomulos_mandibula >= 1.2:
        forma_rostro = "Triangular"
    elif ancho_frente >= ancho_pomulos and ancho_pomulos >= ancho_mandibula:
        forma_rostro = "Cuadrado"
    elif ancho_pomulos > ancho_frente and ancho_pomulos > ancho_mandibula:
        forma_rostro = "Corazón"
    elif ancho_mandibula > ancho_frente and ancho_mandibula > ancho_pomulos:
        forma_rostro = "Diamante"
    else:
        forma_rostro = "Redondo o Alargado"

    return forma_rostro

if __name__ == "__main__":
    imagen = cv2.imread("./rostros/ImagenCapturada.jpg")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Cargar detectores y predictores de puntos de referencia
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./modelos/shape_predictor_81_face_landmarks.dat")
    
    # Detectar rostros
    rostros = detector(gris)

    # Procesar cada rostro detectado
    for rostro in rostros:
        (x, y, w, h) = rostro.left(), rostro.top(), rostro.width(), rostro.height()
        puntos_referencia = predictor(gris, rostro).parts()
        
        # Calcular distancias y clasificar forma del rostro
        distancias = calcular_distancias(puntos_referencia)
        forma_rostro = clasificar_forma_rostro(distancias)
        
        # Dibujar elementos visuales para el rostro
        indice_contorno = list(range(0, 17))  # Incluye del 0 al 16, el contorno del rostro

        # Iterar a través de los puntos del contorno para dibujar líneas
        for i in range(1, len(indice_contorno)):
            # Obtener los puntos de referencia para el contorno
            punto_anterior = puntos_referencia[indice_contorno[i - 1]]
            punto_actual = puntos_referencia[indice_contorno[i]]
            
            # Coordenadas de los puntos
            x1, y1 = punto_anterior.x, punto_anterior.y
            x2, y2 = punto_actual.x, punto_actual.y
            
            # Dibujar línea entre estos puntos del contorno del rostro
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dibujar recuadro y texto para indicar la forma del rostro
        # cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(imagen, f"Forma del rostro: {forma_rostro}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Forma del rostro: {forma_rostro}")
    
    cv2.imshow("Forma del Rostro", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
