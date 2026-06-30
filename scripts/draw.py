import cv2

COLORS = {
    "runway": (0, 255, 255),            # Yellow
    "hangar": (255, 100, 0),            # Blue
    "terminal building": (255, 255, 255)
}


def draw_detections(image, detections):

    output = image.copy()

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        cls = det["class"]

        conf = det["confidence"]

        color = COLORS.get(cls, (0,255,0))

        cv2.rectangle(
            output,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        label = f"{cls} {int(conf*100)}%"

        (w,h),_ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        y = max(h+8, y1)

        cv2.rectangle(
            output,
            (x1,y-h-8),
            (x1+w+6,y),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1+3,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,0,0),
            2
        )

    return output