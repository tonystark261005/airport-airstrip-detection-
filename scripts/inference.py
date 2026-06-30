from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "runs" / "detect" / "airport_detection_v3-3" / "weights" / "best.pt"

# ==========================================================
# MODELS
# ==========================================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(str(MODEL_PATH))

sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(MODEL_PATH),
    confidence_threshold=0.05,
    device=DEVICE
)


# ==========================================================
# Convert detections
# ==========================================================

def convert_results(results):

    detections = []

    names = results.names

    for box in results.boxes:

        cls = names[int(box.cls)]
        conf = float(box.conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({

            "class": cls,

            "confidence": conf,

            "bbox": [x1, y1, x2, y2],

            "center_x": (x1 + x2) // 2,

            "center_y": (y1 + y2) // 2,

            "width": x2 - x1,

            "height": y2 - y1

        })

    return detections


# ==========================================================
# Full image inference
# ==========================================================

def full_image_predict(image):

    results = yolo_model.predict(

        source=image,

        imgsz=1536,

        conf=0.05,

        verbose=False

    )[0]

    return convert_results(results)


# ==========================================================
# Generic SAHI
# ==========================================================

def run_sahi(image, slice_size):

    prediction = get_sliced_prediction(

        image,

        sahi_model,

        slice_height=slice_size,

        slice_width=slice_size,

        overlap_height_ratio=0.35,

        overlap_width_ratio=0.35

    )

    detections = []

    for obj in prediction.object_prediction_list:

        cls = obj.category.name

        conf = float(obj.score.value)

        x1 = int(obj.bbox.minx)
        y1 = int(obj.bbox.miny)
        x2 = int(obj.bbox.maxx)
        y2 = int(obj.bbox.maxy)

        detections.append({

            "class": cls,

            "confidence": conf,

            "bbox": [x1, y1, x2, y2],

            "center_x": (x1 + x2) // 2,

            "center_y": (y1 + y2) // 2,

            "width": x2 - x1,

            "height": y2 - y1

        })

    return detections


# ==========================================================
# Multi-scale SAHI
# ==========================================================

def sahi_predict(image):

    detections = []

    detections.extend(run_sahi(image, 512))
    detections.extend(run_sahi(image, 768))
    detections.extend(run_sahi(image, 1024))

    return detections