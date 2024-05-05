import cv2
import numpy as np
import dlib

# Definir la función para calcular distancias clave entre puntos de referencia
def calcular_distancias(puntos_referencia):
    altura_vertical = np.sqrt(
        (puntos_referencia[15].x - puntos_referencia[1].x) ** 2 + 
        (puntos_referencia[15].y - puntos_referencia[1].y) ** 2
    )
    
    ancho_frente = np.sqrt(
        (puntos_referencia[77].x - puntos_referencia[78].x) ** 2 + 
        (puntos_referencia[77].y - puntos_referencia[78].y) ** 2
    )
    
    ancho_pomulos = np.sqrt(
        (puntos_referencia[3].x - puntos_referencia[8].x) ** 2 + 
        (puntos_referencia[3].y - puntos_referencia[8].y) ** 2
    )
    
    ancho_mandibula = np.sqrt(
        (puntos_referencia[71].x - puntos_referencia[78].x) ** 2 + 
        (puntos_referencia[71].y - puntos_referencia[78].y) ** 2
    )
    
    return altura_vertical, ancho_frente, ancho_pomulos, ancho_mandibula

# Función para clasificar la forma del rostro según las distancias calculadas
def clasificar_forma_rostro(distancias):
    altura_vertical, ancho_frente, ancho_pomulos, ancho_mandibula = distancias
    
    relacion_vertical_frente = altura_vertical / ancho_frente
    relacion_pomulos_mandibula = ancho_pomulos / ancho_mandibula
    
    if relacion_vertical_frente >= 1.3 and relacion_pomulos_mandibula >= 1.0:
        forma_rostro = "Ovalado"
    elif relacion_vertical_frente >= 1.5 and relacion_pomulos_mandibula < 0.8:
        forma_rostro = "Alargado"
    elif relacion_vertical_frente < 1.2 and relacion_pomulos_mandibula >= 1.2:
        forma_rostro = "Triangular"
    elif ancho_frente >= ancho_pomulos and ancho_pomulos >= ancho_mandibula:
        forma_rostro = "Cuadrado"
    elif ancho_pomulos > ancho_frente and ancho_mandibula:
        forma_rostro = "Corazón"
    elif ancho_mandibula > ancho_frente and ancho_pomulos:
        forma_rostro = "Diamante"
    else:
        forma_rostro = "Redondo o Alargado"
    
    return forma_rostro

# Función que clasifica y anota la imagen con la forma del rostro
def clasificar(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./modelos/shape_predictor_81_face_landmarks.dat")
    
    rostros = detector(gris)

    for rostro in rostros:
        puntos_referencia = predictor(gris, rostro).parts()
        
        distancias = calcular_distancias(puntos_referencia)
        forma_rostro = clasificar_forma_rostro(distancias)
        
        # Dibujar el contorno del rostro
        indice_contorno = list(range(0, 17))
        for i in range(1, len(indice_contorno)):
            punto_anterior = puntos_referencia[indice_contorno[i - 1]]
            punto_actual = puntos_referencia[indice_contorno[i]]
            x1, y1 = punto_anterior.x, punto_anterior.y
            x2, y2 = punto_actual.x, punto_actual.y
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Anotar la forma del rostro
        cv2.putText(imagen, f"Forma: {forma_rostro}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return imagen

# Ejemplo de uso de la función
if __name__ == "__main__":
    imagen = cv2.imread("./rostros/rostro4.jpg")
    imagen_anotada = clasificar(imagen)
    cv2.imshow("Rostro Anotado", imagen_anotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
