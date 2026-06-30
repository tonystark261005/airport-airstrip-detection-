def priority_score(d):

    cls = d["class"]

    conf = d["confidence"]

    area = d["width"] * d["height"]

    if cls == "runway":
        score = 100

    elif cls == "terminal building":
        score = 70

    elif cls == "hangar":
        score = 50

    else:
        score = 10

    score += conf * 30

    score += min(area / 15000, 20)

    return round(score, 2)


def priority_level(score):

    if score >= 120:
        return "CRITICAL 🔴"

    elif score >= 90:
        return "HIGH 🟠"

    elif score >= 60:
        return "MEDIUM 🟡"

    return "LOW 🟢"


def prioritize(detections):

    for d in detections:

        score = priority_score(d)

        d["priority"] = score

        d["priority_level"] = priority_level(score)

    detections.sort(
        key=lambda x: x["priority"],
        reverse=True
    )

    return detections