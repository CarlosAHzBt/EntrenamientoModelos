from roboflow import Roboflow
import os

api_key = "H1P5NHM0TQOf2FPAmUsv"
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("obstacle-dmvbe")
model = project.version(1).model

# Directorio principal donde están tus imágenes y subcarpetas
root_directory = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\Imagenes\IMAGENES PARA MODELO v4"

# Directorio para guardar imágenes con confianza 1.0
high_confidence_dir = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\Baches_garantizados"
os.makedirs(high_confidence_dir, exist_ok=True)

# Directorio para guardar imágenes sin confianza 1.0
other_confidence_dir = r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\ImagenesSinBaches"
os.makedirs(other_confidence_dir, exist_ok=True)

# Contadores para nombrar archivos
high_confidence_count = 1
other_confidence_count = 1

# Recorre la carpeta principal y subcarpetas
for subdir, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(subdir, file)
            prediction = model.predict(image_path).json()
            print(prediction)

            # Verifica si alguna predicción tiene una confianza de 1.0
            high_confidence = any(pred['confidence'] == 1.0 for pred in prediction['predictions'])

            # Guarda la imagen en la carpeta correspondiente y actualiza el contador
            if high_confidence:
                output_file = os.path.join(high_confidence_dir, f"{high_confidence_count}.jpg")
                high_confidence_count += 1
            else:
                output_file = os.path.join(other_confidence_dir, f"{other_confidence_count}.jpg")
                other_confidence_count += 1

            model.predict(image_path).save(output_file)

print(f"Imágenes con confianza 1.0: {high_confidence_count - 1}")
print(f"Otras imágenes: {other_confidence_count - 1}")
