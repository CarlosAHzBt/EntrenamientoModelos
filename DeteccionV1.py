from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

# Carga el modelo YOLO
model = YOLO("runs/detect/train5/weights/best.pt")

# Rutas a tus carpetas de imágenes
source_folder = 'Imagenes/IMAGENES PARA MODELO v4/Bache'
target_folder = 'ImagenesDeDeteccionV1/ImagenesConBachesDetectados'

# Asegúrate de que exista la carpeta de destino
os.makedirs(target_folder, exist_ok=True)

# Itera sobre cada archivo en la carpeta
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        frame = cv2.imread(image_path)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realiza la detección
        results = model.predict(img)
        names = model.names

        car_id = list(names)[list(names.values()).index('Bache')]
        number = results[0].boxes.cls.tolist().count(car_id)
        try:
            detections = results[0].boxes.xyxy[0]  # Get detection bounding boxes
        except:
            detections = 0
        annotator = Annotator(frame, line_width=2, font_size=10)
        if number >= 1:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])

            # Guarda la imagen con detecciones
            result_frame = annotator.result()
            cv2.imwrite(os.path.join(target_folder, filename), result_frame)
            # Guardar el txt con las coordenadas de los baches
            # Save coordinates to a txt file
            with open(os.path.join("ImagenesDeDeteccionV1/detected_coords", filename.replace('.png', '.txt')), 'w') as f:
                f.write(str(detections.tolist()))



        else :
            cv2.imwrite(os.path.join("ImagenesDeDeteccionV1/NoIdentificadas", filename), frame)


# Mensaje al finalizar el procesamiento
print("Procesamiento completo, imágenes guardadas en:", target_folder)
