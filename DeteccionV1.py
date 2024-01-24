from ultralytics import YOLO
from PIL import Image
import os
import json
# Load YOLO model
model = YOLO("runs/detect/train5/weights/best.pt")

# Paths to your image folders
source_folder = 'Imagenes/IMAGENES PARA MODELO v4/Bache'
target_folder_high_conf = 'Baches Deteccion En Imagenes.v1i.yolov8'
coords_folder = 'detected_coords'  # Folder to save coordinates

# Ensure target folders exist
os.makedirs(target_folder_high_conf, exist_ok=True)
os.makedirs(coords_folder, exist_ok=True)  # Create folder for coordinates if it doesn't exist

# Iterate over each file in the folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        image = Image.open(image_path)

        # Perform detection
        results = model.predict(source=image, conf=0.5)
        try:
            names = model.names
            car_id = list(names)[list(names.values()).index('Bache')]
            number = results[0].boxes.cls.tolist().count(car_id)

            detections = results[0].boxes.xyxy[0]  # Get detection bounding boxes

            # Check if detections were found
            if number == 1 :
                    # Extract bounding box coordinates
                    for r in results:
                        for c in r.boxes.cls:
                            print(model.names[int(c)])

                    # Save coordinates to a txt file
                    with open(os.path.join(coords_folder, filename.replace('.png', '.txt')), 'w') as f:
                        f.write(str(detections.tolist()))

                    # Optionally save image with high confidence detections
                    image.save(os.path.join(target_folder_high_conf, filename))
        except:
            print("No se detecto nada")
            image.save(os.path.join('ImagenesResultados/NoIdentificadas', filename))


# Print out a message when processing is complete
print("Processing complete, bounding boxes saved to:", coords_folder)
