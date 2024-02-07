from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

# Carga el modelo YOLO
model = YOLO(r"SegmentacionV1\train5\weights\best.pt")

# Rutas a tus carpetas de imágenes
source_folder = r'E:\Imagenes\6FebLey\20240206_073721'
target_folder = r'Imagenes\ImagenesResultadosDeDeteccion\ImagenesConBachesDetectados'
detected_coords_folder = r'Imagenes\ImagenesResultadosDeDeteccion\detected_coords'
no_identificadas_folder = r'Imagenes\ImagenesResultadosDeDeteccion\NoIdentificadas'

# Asegúrate de que existan las carpetas de destino
os.makedirs(target_folder, exist_ok=True)
os.makedirs(detected_coords_folder, exist_ok=True)
os.makedirs(no_identificadas_folder, exist_ok=True)

# Itera sobre todas las carpetas y archivos en la carpeta raíz
for foldername, subfolders, filenames in os.walk(source_folder):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(foldername, filename)
            frame = cv2.imread(image_path)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realiza la detección
        results = model.predict(img)
        with open(os.path.join(detected_coords_folder, filename.replace('.png', '.txt')), 'a') as f:
            for result in results:
                if results[0].boxes is not None and len(results[0].boxes) > 0:  # Verifica si hay detecciones
                    # hacer algo con la detección           
                    annotator = Annotator(frame, line_width=2, font_size=10)
                    for det in result.boxes:  # Itera sobre cada detección
                        b = det.xyxy[0] # Coordenadas de la caja
                        c = int(det.cls)  # Clase de la detección
                        annotator.box_label(b, model.names[c])

                        # Guarda las coordenadas de los baches detectados en un archivo txt
                        #with open(os.path.join(detected_coords_folder, filename.replace('.png', '.txt')), 'a') as f:
                        f.write(f"{b[0]}, {b[1]}, {b[2]}, {b[3]}\n")

                    # Guarda la imagen con detecciones
                    result_frame = annotator.result()
                    cv2.imwrite(os.path.join(target_folder, filename), result_frame)
                else:
                    # Guarda imágenes sin detecciones identificadas en una carpeta específica
                    cv2.imwrite(os.path.join(no_identificadas_folder, filename), frame)

# Mensaje al finalizar el procesamiento
print("Procesamiento completo, imágenes y txt guardados en:", target_folder, detected_coords_folder, no_identificadas_folder)
