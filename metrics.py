

import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

MODEL_PATH = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/runs/detect/train18/weights/best.pt"
IMAGES_DIR = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/processed_images"
LABELS_ABS_DIR = "C:/Users/alexk/OneDrive/Документы/GitHub/proj/labels_abs"
CONF_THRESH = 0.001  
IOU_THRESHOLD = 0.5  

def iou_xyxy(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


def load_ground_truths(images_dir, labels_abs_dir):
    gts = {}
    images = sorted(Path(images_dir).glob("*.*"))

    for img_path in images:
        stem = img_path.stem
        label_path = Path(labels_abs_dir) / f"{stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x1, y1, x2, y2 = map(float, parts[:4])
                        cls = int(parts[4]) if len(parts) == 5 else 0
                        boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls})
        gts[img_path.name] = boxes
    return gts


def run_inference(model, images_dir, conf=0.001):
    preds = {}
    images = sorted(Path(images_dir).glob("*.*"))

    for img_path in tqdm(images, desc="Inference"):
        results = model(str(img_path), conf=conf, verbose=False)
        res = results[0]
        boxes = res.boxes

        pred = []
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf_score = float(b.conf.item())
            cls = int(b.cls.item())
            pred.append([x1, y1, x2, y2, conf_score, cls])

        preds[img_path.name] = pred
    return preds


def compute_metrics(preds, gts, iou_thr=0.5):
    TP = 0
    FP = 0
    FN = 0

    for img_name, gt_boxes in gts.items():
        pred_boxes = preds.get(img_name, [])

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        for pb in pred_boxes:
            pred_box = pb[:4]
            ious = [iou_xyxy(pred_box, gt["bbox"]) for gt in gt_boxes]
            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] >= iou_thr and not matched_gt[max_iou_idx]:
                    TP += 1
                    matched_gt[max_iou_idx] = True
                else:
                    FP += 1
            else:
                FP += 1

        FN += np.sum(~matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    print(f"Загружаем модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("Загружаем ground-truth разметку...")
    gts = load_ground_truths(IMAGES_DIR, LABELS_ABS_DIR)

    print("Запускаем инференс на GPU (если доступен)...")
    preds = run_inference(model, IMAGES_DIR, conf=CONF_THRESH)

    print("Считаем Precision, Recall и F1-score при IoU=0.5...")
    metrics = compute_metrics(preds, gts, iou_thr=IOU_THRESHOLD)

    print("\n====== РЕЗУЛЬТАТЫ ======")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
