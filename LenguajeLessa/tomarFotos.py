import cv2
import os
import time

# Configura el número de fotos a tomar
desde = 200
numero_de_fotos = 250 # Puedes cambiar este número
carpeta_destino = 'C:\\Users\\juanc\\OneDrive\\Documentos\\ImagenesLessa\\O'


# Crea la carpeta si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Verifica si la cámara está abierta
if not cap.isOpened():
    print("No se puede abrir la cámara")
else:
    for i in range(desde, desde + numero_de_fotos):
        # Captura el cuadro
        ret, frame = cap.read()
        
        if not ret:
            print("No se puede recibir frame (la cámara está desconectada o hubo un error)")
            break

        # Guarda la imagen en el directorio de destino
        nombre_archivo = os.path.join(carpeta_destino, f'O_{i+1}.jpg')
        cv2.imwrite(nombre_archivo, frame)
        
        print(f'Foto {i+1} guardada como {nombre_archivo}')
        time.sleep(0.1)

    # Libera la cámara y cierra cualquier ventana de OpenCV
    cap.release()
    cv2.destroyAllWindows()
