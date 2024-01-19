from ultralytics import YOLO
from PIL import Image
import os

# Load YOLO model
model = YOLO("runs/classify/train9/weights/best.pt")

# Paths to your image folders
source_folder = 'Imagenes/IMAGENES PARA MODELO v4/Nada'
target_folder_high_conf = 'BachesPrueba2'
target_folder_low_conf = 'PrediccionesBajas'  # Carpeta para confianza baja

# Ensure target folders exist
os.makedirs(target_folder_high_conf, exist_ok=True)
os.makedirs(target_folder_low_conf, exist_ok=True)

# Iterate over each file in the folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        image = Image.open(image_path)

        # Perform detection
        results = model.predict(source=image, conf=0.5)
        DeteccionGanadora = results[0].probs.top1
        ConfianzaGanadora = results[0].probs.top1conf.item()

        # Check if detections were found
        if ConfianzaGanadora >= 0.5:
            if DeteccionGanadora == 1:
                image.save(os.path.join(target_folder_high_conf, filename))
            elif DeteccionGanadora == 0:
                image.save(os.path.join('ImagenesResultados/CalleBien', filename))
            elif DeteccionGanadora == 2:
                image.save(os.path.join('ImagenesResultados/Grietas', filename))
        else:
            # If the top prediction is below the confidence threshold, save to a separate folder
            image.save(os.path.join('ImagenesResultados/NoIdentificadas', filename))
