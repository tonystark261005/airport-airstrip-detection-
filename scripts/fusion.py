import math


def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0

    inter = (x2 - x1) * (y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (area1 + area2 - inter)

def fuse_detections(yolo_det, sahi_det, threshold=0.25):

    merged = []

    all_det = yolo_det + sahi_det

    used = set()

    for i, det in enumerate(all_det):

        if i in used:
            continue

        best = det

        for j in range(i+1, len(all_det)):

            if j in used:
                continue

            other = all_det[j]

            if det["class"] != other["class"]:
                continue

            if iou(det["bbox"], other["bbox"]) > threshold:

                if other["confidence"] > best["confidence"]:
                    best = other

                used.add(j)

        merged.append(best)

    return merged