from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    print("CUDA está disponible. GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA no está disponible.")

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n-cls.pt")

    model.train(data=r"C:\Users\carlo\PycharmProjects\PruebaSegmentacion\Imagenes\IMAGENES PARA MODELO v4\Organizado",
                epochs=50, batch=8, imgsz=848)

if __name__ == '__main__':
    train_model()
