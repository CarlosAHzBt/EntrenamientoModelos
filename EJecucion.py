from ultralytics import YOLO
from PIL import Image
import os

# Cargar el modelo YOLO
model = YOLO("runs/classify/train9/weights/best.pt")

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Imagenes/IMAGENES PARA MODELO v4/Nada'

# Iterar sobre cada archivo en la carpeta
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Asegúrate de que es una imagen
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # Realizar la detección
        results = model.predict(source=image)

        # Inspeccionar la estructura de 'results'
        print(type(results))  # Ver el tipo de objeto que es 'results'
        print(results)  # Ver el contenido de 'results'

        # [Aquí irá el código para procesar los resultados una vez que entiendas su estructura]
