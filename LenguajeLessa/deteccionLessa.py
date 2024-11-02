import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modelo_vocales.keras")

# Etiquetas de las clases
class_names = ['A', 'E']
confidence_threshold = 0.9  # Umbral de confianza

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer imagen de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar la imagen para el modelo
    img = cv2.resize(frame, (64, 64))  # Cambia el tamaño al tamaño de entrada del modelo
    img = img / 255.0                  # Normalizar
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch

    # Realizar predicción
    predictions = model.predict(img)
    confidence = np.max(predictions)  # Obtener la confianza de la predicción más alta
    if confidence > confidence_threshold:
        predicted_class = class_names[np.argmax(predictions)]
    else:
        predicted_class = "Letra no detectada"

    # Mostrar la predicción en la ventana de video
    cv2.putText(frame, f"Prediccion: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detección de Vocales", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
