from ultralytics import YOLO
from PIL import Image
import os

# Load YOLO model
model = YOLO(r"/media/mcc/ELITE SE880/Modelo Clasificacion/ModeloV4/train17/weights/best.pt")

# Paths to your image folders
source_folder = r'/media/mcc/ELITE SE880/8feb/imagenes'
ruta_no_identificadas = r'/media/mcc/ELITE SE880/8feb/ModeloV4/imagenesResultados/NoIdentificadas'
target_folder_high_conf = r'/media/mcc/ELITE SE880/8feb/ModeloV4/imagenesResultados/Baches'
ruta_calle_bien = r'/media/mcc/ELITE SE880/8feb/ModeloV4/imagenesResultados/CalleBien'
ruta_grietas = r'/media/mcc/ELITE SE880/8feb/ModeloV4/imagenesResultados/Grietas' 
# Ensure target folders exist
os.makedirs(target_folder_high_conf, exist_ok=True)
os.makedirs(ruta_calle_bien , exist_ok=True)
os.makedirs(ruta_grietas , exist_ok=True)
os.makedirs(ruta_no_identificadas , exist_ok=True)

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
    image.save(os.path.join(folder, f"{counters[category]}.png"))
    counters[category] += 1

# Iterate over each file in the folder and subfolders
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)

            # Perform detection
            results = model.predict(source=image, conf=0.5)
            DeteccionGanadora = results[0].probs.top1
            ConfianzaGanadora = results[0].probs.top1conf.item()

            # Check if detections were found
            if ConfianzaGanadora >= 0.1:
                if DeteccionGanadora == 0:
                    save_image(image, target_folder_high_conf, "Baches")
                elif DeteccionGanadora == 2:
                    save_image(image, ruta_calle_bien , "CalleBien")
                elif DeteccionGanadora == 1:
                    save_image(image, ruta_grietas, "Grietas")
            else:
                # If the top prediction is below the confidence threshold, save to a separate folder
                save_image(image, ruta_no_identificadas, "NoIdentificadas")
