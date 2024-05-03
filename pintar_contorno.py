import cv2
import dlib

# Cargar el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el predictor de puntos faciales de dlib
predictor = dlib.shape_predictor("./modelos/shape_predictor_68_face_landmarks.dat")

# Cargar la imagen
image = cv2.imread('./rostros/rostro2.png')

# Convertir a escala de grises para facilitar la detección
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar rectángulos y obtener contornos
for (x, y, w, h) in faces:
    # Dibujar rectángulo alrededor del rostro
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Obtener puntos faciales para contorno del rostro
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(gray, rect)

    # Dibujar el contorno del rostro usando los landmarks faciales
    for i in range(1, 17):  # Los puntos 1 a 16 forman el contorno del rostro
        pt1 = (shape.part(i - 1).x, shape.part(i - 1).y)
        pt2 = (shape.part(i).x, shape.part(i).y)
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)  # Línea roja para el contorno del rostro

# Mostrar la imagen con los contornos dibujados
cv2.imshow("Contornos del rostro", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
