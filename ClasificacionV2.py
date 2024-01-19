from ultralytics import YOLO
from PIL import Image
import os

# Load YOLO model
model = YOLO(r"runs\classify\train3\weights\best.pt")

# Paths to your image folders
source_folder = r'C:\Users\PC\Documents\PuntoMinimo\Datos_Extraccion'
target_folder_high_conf = 'ImagenesDeLaClasificacion/Baches'

# Ensure target folders exist
os.makedirs(target_folder_high_conf, exist_ok=True)
os.makedirs('ImagenesDeLaClasificacion/CalleBien' , exist_ok=True)
os.makedirs('ImagenesDeLaClasificacion/Grietas' , exist_ok=True)
os.makedirs('ImagenesDeLaClasificacion/NoIdentificadas' , exist_ok=True)

# Counters for each classification
counters = {
    "Baches": 1,
    "CalleBien": 1,
    "Grietas": 1,
    "NoIdentificadas": 1
}

# Function to save image with new name
def save_image(image, folder, category):
    global counters
    image.save(os.path.join(folder, f"{counters[category]}.jpg"))
    counters[category] += 1

# Iterate over each file in the folder and subfolders
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)

            # Perform detection
            results = model.predict(source=image, conf=0.9)
            DeteccionGanadora = results[0].probs.top1
            ConfianzaGanadora = results[0].probs.top1conf.item()

            # Check if detections were found
            if ConfianzaGanadora >= 0.5:
                if DeteccionGanadora == 0:
                    save_image(image, target_folder_high_conf, "Baches")
                elif DeteccionGanadora == 2:
                    save_image(image, 'ImagenesDeLaClasificacion/CalleBien', "CalleBien")
                elif DeteccionGanadora == 1:
                    save_image(image, 'ImagenesDeLaClasificacion/Grietas', "Grietas")
            else:
                # If the top prediction is below the confidence threshold, save to a separate folder
                save_image(image, 'ImagenesDeLaClasificacion/NoIdentificadas', "NoIdentificadas")
