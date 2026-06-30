import numpy as np

from scripts.inference import (
    full_image_predict,
    sahi_predict
)

from scripts.fusion import fuse_detections
from scripts.postprocess import postprocess
from scripts.prioritize import prioritize


def detect_objects(rgb_image):

    if not isinstance(rgb_image, np.ndarray):
        rgb_image = np.array(rgb_image)

    # -----------------------------------------
    # Full-image YOLO
    # -----------------------------------------

    yolo_detections = full_image_predict(rgb_image)

    # -----------------------------------------
    # SAHI
    # -----------------------------------------

    sahi_detections = sahi_predict(rgb_image)

    # -----------------------------------------
    # Fusion
    # -----------------------------------------

    detections = fuse_detections(
        yolo_detections,
        sahi_detections
    )

    print("\n========== BEFORE POSTPROCESS ==========\n")

    for d in detections:
        if d["class"] == "runway":
            print(d)

    # -----------------------------------------
    # Postprocess
    # -----------------------------------------

    detections = postprocess(detections)

    print("\n========== AFTER POSTPROCESS ==========\n")

    for d in detections:
        if d["class"] == "runway":
            print(d)

    # -----------------------------------------
    # Priority
    # -----------------------------------------

    detections = prioritize(detections)

    return rgb_image.copy(), detections