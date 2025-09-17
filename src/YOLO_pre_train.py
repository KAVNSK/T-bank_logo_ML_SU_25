import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import torch
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        print("CUDA доступна")
    else:
        print("CUDA не доступна")

    model = YOLO('yolov8n.pt')

    model.train(
        data=f'C:/Users/alexk/OneDrive/Документы/GitHub/proj/src/data.yaml',
        epochs=20,
        imgsz=640,
        device='0',
        batch=4,
    )

    result = model.predict(
        "C:/Users/alexk/OneDrive/Документы/GitHub/proj/dataset_full/data_sirius/c774b784e28f257484be58eb16347d4b.jpg",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device='0'
    )

    # Вывод через OpenCV
    import cv2
    img = result[0].plot()
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
