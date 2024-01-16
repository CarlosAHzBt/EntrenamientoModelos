import os
import shutil


def obtener_archivos_en_carpeta(carpeta):
    """ Obtiene una lista de todos los archivos en una carpeta, incluyendo subcarpetas. """
    archivos = []
    for subdir, dirs, files in os.walk(carpeta):
        for file in files:
            archivos.append(os.path.relpath(os.path.join(subdir, file), carpeta))
    return set(archivos)


# Rutas a las dos carpetas para comparar
carpeta1 = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\Baches_garantizados"
carpeta2 = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\Imagenes"

# Ruta a la carpeta donde se guardarán los archivos que no están en ambas carpetas
carpeta_nueva = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\ImagenesSinBaches"
os.makedirs(carpeta_nueva, exist_ok=True)

# Obtiene los nombres de los archivos en ambas carpetas
archivos_carpeta1 = obtener_archivos_en_carpeta(carpeta1)
archivos_carpeta2 = obtener_archivos_en_carpeta(carpeta2)

# Filtra los archivos que no están en carpeta1
archivos_unicos = archivos_carpeta2 - archivos_carpeta1

# Copia los archivos únicos a la nueva carpeta
for archivo in archivos_unicos:
    ruta_original = os.path.join(carpeta2, archivo)
    ruta_destino = os.path.join(carpeta_nueva, archivo)
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)  # Crea subdirectorios si es necesario
    shutil.copy2(ruta_original, ruta_destino)

print(f"Archivos copiados a {carpeta_nueva}: {len(archivos_unicos)}")
