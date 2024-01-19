from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    print("CUDA está disponible. GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA no está disponible.")

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n-cls.pt")

    model.train(data=r"C:\Users\PC\Desktop\Carlos\IMAGENES PARA MODELO\Imagenes para modelo V5",
                epochs=500, batch=2, imgsz=848)

if __name__ == '__main__':
    train_model()
