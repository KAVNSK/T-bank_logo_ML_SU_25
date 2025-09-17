import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
from PIL import Image

# Параметры
INPUT_DIR = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_post_YOLO/images" # Папка с исходными изображениями
OUTPUT_IMAGES_DIR = "processed_images" # Папка для всех обработанных изображений
LABELS_ABS_DIR = "labels_abs"         # Папка для абсолютных координат
LABELS_YOLO_DIR = "labels_yolo"       # Папка для YOLO координат
LABELS_NO_LOGO_DIR = "labels_no_logo" # Папка для изображений без логотипа
YOLO_MODEL_PATH = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/runs/detect/train18/weights/best.pt"# Путь к вашей предобученной модели YOLO

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_ABS_DIR, exist_ok=True)
os.makedirs(LABELS_YOLO_DIR, exist_ok=True)
os.makedirs(LABELS_NO_LOGO_DIR, exist_ok=True)

model = YOLO("C:/Users/alexk/OneDrive/Документы/GitHub/proj/runs/detect/train18/weights/best.pt")

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

for img_path in Path(INPUT_DIR).glob("*.*"):
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    results = model(img_path)
    detections = results[0].boxes  # YOLOv8 формат
    
    shutil.copy(img_path, os.path.join(OUTPUT_IMAGES_DIR, img_path.name))
    
    if len(detections) > 0:
        abs_label_file = os.path.join(LABELS_ABS_DIR, img_path.stem + ".txt")
        yolo_label_file = os.path.join(LABELS_YOLO_DIR, img_path.stem + ".txt")
        
        with open(abs_label_file, "w") as f_abs, open(yolo_label_file, "w") as f_yolo:
            for det in detections:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                cls = int(det.cls.item())
                
                # абсолютные координаты
                f_abs.write(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {cls}\n")
                
                # YOLO формат
                x_c, y_c, w, h = convert_to_yolo(x1, y1, x2, y2, img_w, img_h)
                f_yolo.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    else:
        no_logo_file = os.path.join(LABELS_NO_LOGO_DIR, img_path.stem + ".txt")
        with open(no_logo_file, "w") as f:
            f.write("0\n")

print("Обработка завершена")
