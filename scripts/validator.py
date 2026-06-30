import cv2

def validate_image(image):

    checks = []
    status = "green"

    h, w = image.shape[:2]

    # Resolution (only reject ridiculously small images)

    if min(h, w) < 200:
        checks.append(("🔴", "Image resolution is extremely low"))
        status = "red"
    else:
        checks.append(("🟢", "Resolution OK"))

    # Blur (warning only)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 20:
        checks.append(("🟡", "Image appears blurry"))
        if status != "red":
            status = "yellow"
    else:
        checks.append(("🟢", "Image quality OK"))

    # RGB check

    if image.ndim != 3 or image.shape[2] != 3:
        checks.append(("🔴", "Invalid image"))
        status = "red"
    else:
        checks.append(("🟢", "Valid RGB image"))

    return status, checks