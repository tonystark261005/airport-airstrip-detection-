from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np

class SahiRunner:
    def __init__(self, model_path, device="cuda:0"):
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.25,
            device=device
        )

    def predict(self, image):
        """
        SAHI tiled inference (fixes runway miss issue)
        """
        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3
        )
        return result


    def extract_boxes(self, result):
        """
        Convert SAHI output → clean bbox format
        """
        detections = []

        for obj in result.object_prediction_list:
            box = obj.bbox

            detections.append({
                "class": obj.category.name,
                "confidence": float(obj.score.value),
                "x1": int(box.minx),
                "y1": int(box.miny),
                "x2": int(box.maxx),
                "y2": int(box.maxy)
            })

        return detections