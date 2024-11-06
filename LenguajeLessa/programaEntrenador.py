import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Rutas del dataset
dataset_dir = "C:\\Users\\juanc\\OneDrive\\Documentos\\ImagenesLessa"
img_size = (64, 64)
batch_size = 32

# Preprocesamiento de datos: crea lotes de datos de entrenamiento y validación
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Definir el modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 salidas para las letras A E, I y O
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Guarda el modelo en formato .keras
model.save("modelo_vocales.keras")
