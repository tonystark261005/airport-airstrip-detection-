import numpy as np
from PIL import Image

from scripts.geotiff import read_geotiff


def load_image(uploaded_files):

    """
    Returns

    rgb
    transform
    crs
    is_geotiff
    """

    transform = None
    crs = None
    is_geotiff = False

    # -----------------------------
    # JPG / PNG
    # -----------------------------

    if len(uploaded_files) == 1:

        name = uploaded_files[0].name.lower()

        if name.endswith((".jpg", ".jpeg", ".png")):

            image = Image.open(uploaded_files[0]).convert("RGB")

            rgb = np.array(image)

            return rgb, transform, crs, False

    # -----------------------------
    # GeoTIFF
    # -----------------------------

    if len(uploaded_files) == 1:

        rgb, transform, metadata = read_geotiff(uploaded_files[0])

        crs = metadata.get("crs")

        return rgb, transform, crs, True

    # -----------------------------
    # Sentinel B02+B03+B04
    # -----------------------------

    if len(uploaded_files) == 3:

        rgb, transform, metadata = read_geotiff(uploaded_files)

        crs = metadata.get("crs")

        return rgb, transform, crs, True

    raise Exception(
        "Upload either:\n\n"
        "- JPG/PNG\n"
        "- One GeoTIFF\n"
        "- OR B02+B03+B04"
    )