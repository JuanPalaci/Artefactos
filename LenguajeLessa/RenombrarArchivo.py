import os

def renombrar_fotos(direccion_carpeta):
    # Lista todos los archivos en la carpeta
    archivos = os.listdir(direccion_carpeta)
    
    # Filtra solo archivos con extensión de imagen
    fotos = [archivo for archivo in archivos if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Renombra cada foto al formato A_x
    for indice, foto in enumerate(fotos, start=1):
        # Obtiene la extensión de la foto
        extension = os.path.splitext(foto)[1]
        
        # Nuevo nombre con el formato deseado
        nuevo_nombre = f"U_{indice}{extension}"
        
        # Ruta completa de los archivos
        ruta_original = os.path.join(direccion_carpeta, foto)
        ruta_nueva = os.path.join(direccion_carpeta, nuevo_nombre)
        
        # Renombra el archivo
        os.rename(ruta_original, ruta_nueva)
        print(f"Renombrado: {foto} -> {nuevo_nombre}")

# Ejemplo de uso
direccion_carpeta = "C:\\Users\\juanc\\Downloads\\u\\u"
renombrar_fotos(direccion_carpeta)
