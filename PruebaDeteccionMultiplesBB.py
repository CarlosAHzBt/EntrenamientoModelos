#Codigo para probar en una imagen multiples detecciones

from ultralytics import YOLO
import cv2

# Carga el modelo YOLOv8 preentrenado
model = YOLO(r'SegmentacionV1\train5\weights\best.pt')

# Carga una imagen (reemplaza 'ruta/a/tu/imagen.jpg' con la ubicación de tu imagen)
image_path = r'Imagenes\ImagenesResultadosDeDeteccion\ImagenesConBachesDetectados\frame_00062.png'
img = cv2.imread(image_path)

# Realiza la detección en la imagen
results = model.predict(img)

# Abre un archivo de texto para guardar los bounding boxes
with open('bounding_boxes.txt', 'w') as txt_file:
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]  # Obtiene las coordenadas del bounding box (izquierda, arriba, derecha, abajo)
            c = int(box.cls)  # Obtiene la clase del objeto detectado
            txt_file.write(f"{b[0]}, {b[1]}, {b[2]}, {b[3]}\n")  # Escribe en el archivo de texto
            


print("Bounding boxes guardados en bounding_boxes.txt")
