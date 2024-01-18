from ultralytics import YOLO
from PIL import Image
import os

# Cargar el modelo YOLO
model = YOLO("runs/classify/train9/weights/best.pt")

# Ruta a la carpeta que contiene las imágenes
source_folder = 'Imagenes/IMAGENES PARA MODELO v4/Nada'
target_folder = 'BachesPrueba2'

# Asegúrate de que la carpeta de destino existe
os.makedirs(target_folder, exist_ok=True)

# Obtener el índice de la clase 'bache'
bache_class_id = None
for id, name in model.names.items():
    if name == 'bache':
        bache_class_id = id
        break

if bache_class_id is None:
    raise ValueError("Clase 'bache' no encontrada en el modelo")

# Iterar sobre cada archivo en la carpeta
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        image = Image.open(image_path)

        # Realizar la detección
        results = model.predict(source=image, conf=0.5)
        DeteccionGanadaros = results[0].probs.top1
        # Verificar si se encontraron detecciones
        if DeteccionGanadaros==1:
                if bache_class_id:
                    image.save(os.path.join(target_folder, filename))
        if DeteccionGanadaros==0:
                    image.save(os.path.join('CalleBien', filename))

        if DeteccionGanadaros==2:
                    image.save(os.path.join('Grietas', filename))