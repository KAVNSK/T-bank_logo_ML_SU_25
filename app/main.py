from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import tempfile
import os
import shutil
from ultralytics import YOLO

# ---------- Pydantic models (с твоего контракта) ----------
class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

# ---------- FastAPI app ----------
app = FastAPI(title="Logo Detection API")

# Модель загружается один раз при старте приложения
@app.on_event("startup")
def load_model():
    model_path = os.getenv("MODEL_PATH", "C:/Users/alexk/OneDrive/Документы/GitHub/proj/runs/detect/train18/weights/best.pt")
    if not Path(model_path).exists():
        # логируем, но не падаем — модель может быть на volume позже
        print(f"[warning] MODEL_PATH {model_path} does not exist at startup.")
    # загрузка (ultralytics YOLO)
    app.state.model = YOLO(model_path)
    print("Model loaded from", model_path)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@app.post("/detect", response_model=DetectionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа на изображении (возвращает абсолютные координаты)
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        return JSONResponse(status_code=400, content={"error":"bad_request", "detail": f"Unsupported file type: {suffix}"})

    # сохраняем файл во временную папку
    tmpdir = Path(tempfile.mkdtemp())
    tmp_path = tmpdir / filename
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)

        # инференс
        results = app.state.model(str(tmp_path))
        res = results[0]
        detections = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            # извлекаем массивы
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = res.boxes.xyxy
            try:
                confs = res.boxes.conf.cpu().numpy()
            except Exception:
                confs = res.boxes.conf
            try:
                clss = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                clss = res.boxes.cls

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(float, xyxy[i].tolist())
                # cast to ints for API
                bbox = BoundingBox(x_min=int(round(x1)), y_min=int(round(y1)),
                                   x_max=int(round(x2)), y_max=int(round(y2)))
                detections.append(Detection(bbox=bbox))

        return DetectionResponse(detections=detections)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"internal_error", "detail": str(e)})
    finally:
        # cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
