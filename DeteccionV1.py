from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

# Carga el modelo YOLO
model = YOLO(r'C:\Users\carlo\PycharmProjects\pythonProject4\EntrenamientoModelos\Modelos\Deteccion\ModeloV3\weights\best.pt')

# Rutas a tu carpeta raíz de imágenes
root_folder = r'E:\Imagenes\Imagenes - copia'

# Asegúrate de que exista la carpeta de destino para imágenes y txt
target_folder_images = r'E:\Resultados Por Versiones De los modelos, Imagenes\ImgenesDeteccionPorModelos\ModeloV4\Bache'
target_folder_images_without_bounding_box = r'E:\Resultados Por Versiones De los modelos, Imagenes\ImgenesDeteccionPorModelos\ModeloV4\BacheSinBBox'
target_folder_txt = r'E:\Resultados Por Versiones De los modelos, Imagenes\ImgenesDeteccionPorModelos\ModeloV4\Coordenadas'
target_folder_no_identificadas = r'E:\Resultados Por Versiones De los modelos, Imagenes\ImgenesDeteccionPorModelos\ModeloV4\NoIdentificadas'

os.makedirs(target_folder_images, exist_ok=True)
os.makedirs(target_folder_images_without_bounding_box, exist_ok=True)
os.makedirs(target_folder_txt, exist_ok=True)
os.makedirs(target_folder_no_identificadas, exist_ok=True)

# Itera sobre todas las carpetas y archivos en la carpeta raíz
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(foldername, filename)
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
            annotator = Annotator(frame, line_width=4, font_size=10)
            if number >= 1:
                # Guardar la imagen sin detecciones
                output_folder_images_without_bounding_box = os.path.join(target_folder_images_without_bounding_box,
                                                                         os.path.relpath(foldername, root_folder))
                os.makedirs(output_folder_images_without_bounding_box, exist_ok=True)
                output_path_images_without_bounding_box = os.path.join(output_folder_images_without_bounding_box,
                                                                       filename)
                cv2.imwrite(output_path_images_without_bounding_box, frame)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])

                # Guarda la imagen con detecciones
                result_frame = annotator.result()
                output_folder_images = os.path.join(target_folder_images, os.path.relpath(foldername, root_folder))
                os.makedirs(output_folder_images, exist_ok=True)
                output_path_images = os.path.join(output_folder_images, filename)
                cv2.imwrite(output_path_images, result_frame)




                # Guardar el txt con las coordenadas de los baches
                output_folder_txt = os.path.join(target_folder_txt, os.path.relpath(foldername, root_folder))
                os.makedirs(output_folder_txt, exist_ok=True)
                output_path_txt = os.path.join(output_folder_txt, filename.replace('.png', '.txt'))
                with open(output_path_txt, 'w') as f:
                    f.write(str(detections.tolist()))

            else:
                output_folder_no_identificadas = os.path.join(target_folder_no_identificadas, os.path.relpath(foldername, root_folder))
                os.makedirs(output_folder_no_identificadas, exist_ok=True)
                output_path_no_identificadas = os.path.join(output_folder_no_identificadas, filename)
                cv2.imwrite(output_path_no_identificadas, frame)

# Mensaje al finalizar el procesamiento
print("Procesamiento completo, imágenes y txt guardados en:", target_folder_images, target_folder_txt, target_folder_no_identificadas)
