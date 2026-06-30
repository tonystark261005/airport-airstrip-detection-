def aspect_ratio(width, height):
    if min(width, height) == 0:
        return 999
    return max(width, height) / min(width, height)


def area(width, height):
    return width * height


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0

    return inter / union


def merge_boxes(a, b):

    x1 = min(a["bbox"][0], b["bbox"][0])
    y1 = min(a["bbox"][1], b["bbox"][1])
    x2 = max(a["bbox"][2], b["bbox"][2])
    y2 = max(a["bbox"][3], b["bbox"][3])

    return {
        "class": "runway",
        "confidence": max(a["confidence"], b["confidence"]),
        "bbox": [x1, y1, x2, y2],
        "center_x": (x1 + x2) // 2,
        "center_y": (y1 + y2) // 2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def stitch_runways(runways):

    changed = True

    while changed:

        changed = False
        merged = []
        used = [False] * len(runways)

        for i in range(len(runways)):

            if used[i]:
                continue

            current = runways[i]

            for j in range(i + 1, len(runways)):

                if used[j]:
                    continue

                if iou(current["bbox"], runways[j]["bbox"]) > 0.10:
                    current = merge_boxes(current, runways[j])
                    used[j] = True
                    changed = True

            merged.append(current)

        runways = merged

    return runways


def postprocess(detections):

    filtered = []

    for d in detections:

        cls = d["class"]
        conf = d["confidence"]
        w = d["width"]
        h = d["height"]

        ar = aspect_ratio(w, h)
        a = area(w, h)

        if cls == "runway":

            if conf < 0.10:
                continue

            if ar < 2.2:
                continue

            if a < 1500:
                continue

        elif cls == "hangar":

            if conf < 0.35:
                continue

            if a < 500:
                continue

        elif cls == "terminal building":

            if conf < 0.35:
                continue

            if a < 700:
                continue

        filtered.append(d)

    filtered.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    final = []

    while filtered:

        best = filtered.pop(0)

        final.append(best)

        remaining = []

        for obj in filtered:

            if obj["class"] != best["class"]:
                remaining.append(obj)
                continue

            if iou(best["bbox"], obj["bbox"]) < 0.45:
                remaining.append(obj)

        filtered = remaining

    runways = [d for d in final if d["class"] == "runway"]
    others = [d for d in final if d["class"] != "runway"]

    runways = stitch_runways(runways)

    return others + runways