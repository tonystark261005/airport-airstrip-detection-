from rasterio.transform import xy


def get_pixel_center(det):

    return (
        det["center_x"],
        det["center_y"]
    )


def get_latlon(det, transform):

    if transform is None:
        return None

    lon, lat = xy(
        transform,
        det["center_y"],
        det["center_x"]
    )

    return (

        round(lat, 6),

        round(lon, 6)

    )


def attach_coordinates(detections, transform=None):

    output = []

    for det in detections:

        det = det.copy()

        det["pixel"] = get_pixel_center(det)

        if transform is not None:

            det["latlon"] = get_latlon(
                det,
                transform
            )

        output.append(det)

    return output