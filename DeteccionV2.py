#Codigo para hacer la deteccion de un conjunto de imagenes que se encuentran en una carpeta y subcarpetas
#Y guardar las imagenes que tengan la deteccion de un bache en una carpeta, si tienen grietas en otra, si no se detecta nada en otra y si la calle esta bien en otra
#Se dara prioridad a la de baches, si se detecta un bache, la imagen se guardara en la carpeta de baches, si no se detecta un bache pero si grietas, se guardara en la carpeta de grietas
#Si no se detecta nada, se guardara en la carpeta de no identificadas y si la calle esta bien, se guardara en la carpeta de calle bien
#Si se detecta baches y grietas en la misma imagen, se guardara en la carpeta de baches
#Si se detecta baches y calle bien en la misma imagen, se guardara en la carpeta de baches
#Si se detecta grietas y calle bien en la misma imagen, se guardara en la carpeta de grietas
#Si se detecta baches, grietas y calle bien en la misma imagen, se guardara en la carpeta de baches
#Si no se detecta nada, se guardara en la carpeta de no identificadas

from ultralytics import YOLO
from PIL import Image
import os
import cv2
from ultralytics.utils.plotting import Annotator


# Load YOLO model deteccion
model = YOLO(r"SegmentacionV1/train5/weights/best.pt")

# Paths to your image folders
source_folder = r'/media/mcc/ELITE SE880/12feb/imagenes'
ruta_no_identificadas = r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/NoIdentificadas'
target_folder_high_conf = r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/Baches'
ruta_calle_bien = r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/CalleBien'
ruta_grietas = r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/Grietas'
ruta_bacheBB = r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/BachesBB'
# Ensure target folders exist
os.makedirs(target_folder_high_conf, exist_ok=True)
#os.makedirs(ruta_calle_bien , exist_ok=True)
#os.makedirs(ruta_grietas , exist_ok=True)
os.makedirs(ruta_no_identificadas , exist_ok=True)
os.makedirs(ruta_bacheBB, exist_ok=True)

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
            frame = cv2.imread(image_path)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform detection
            results = model.predict(source=img, conf=0.5)
            names = model.names
            car_id = list(names)[list(names.values()).index('Bache')]
            number = results[0].boxes.cls.tolist().count(car_id)
            try:
                detections = results[0].boxes.xyxy[0]  # Get detection bounding boxes
            except:
                detections = 0
            anotator = Annotator(frame, line_width=2, font_size=10)
            #confidence = results[0].boxes.  # Get confidence of top prediction
            # Check if detections were found
            if number >= 1:
                confidence = results[0].boxes[0].conf
                # If the top prediction is above the confidence threshold, save to a separate folder
                if confidence > 0.7:
                    save_image(Image.open(image_path), target_folder_high_conf, "Baches")
                    #Guardar en otra carpeta la imagen con las detecciones
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            b = box.xyxy[0]
                            c = box.cls
                            anotator.box_label(b, model.names[int(c)])
                        result_frame = anotator.result()
                        cv2.imwrite(os.path.join(r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/BachesBB', filename), result_frame)
                    cv2.imwrite(os.path.join(r'/media/mcc/ELITE SE880/12feb/Deteccion/imagenesResultados/BachesBB', filename), result_frame)
                #ave_image(Image.open(image_path), r'/media/mcc/ELITE SE880/8feb/Deteccion/imagenesResultados/BachesBB',  "Baches")

            else:
                # If the top prediction is below the confidence threshold, save to a separate folder
                save_image(Image.open(image_path), ruta_no_identificadas, "NoIdentificadas")
            
# Mensaje al finalizar el procesamiento
print("Procesamiento completo, imágenes guardadas en:", target_folder_high_conf)
print("Procesamiento completo, imágenes guardadas en:", ruta_no_identificadas)