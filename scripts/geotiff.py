import rasterio
import numpy as np


def normalize_band(band):
    """
    Percentile normalization for satellite imagery.
    """

    band = band.astype(np.float32)

    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)

    if p98 - p2 == 0:
        return np.zeros_like(band, dtype=np.uint8)

    band = np.clip(band, p2, p98)
    band = (band - p2) / (p98 - p2)

    return (band * 255).astype(np.uint8)


def read_geotiff(source):
    """
    Universal GeoTIFF Reader

    Supports:

    1. Single GeoTIFF
    2. RGB GeoTIFF
    3. Separate Sentinel bands (B02, B03, B04)

    Returns
    -------
    rgb
    transform
    metadata
    """

    # =====================================================
    # CASE 1 : Three separate TIFF files
    # =====================================================

    if isinstance(source, list):

        bands = {}

        for file in source:

            name = file.name.lower()

            if "b02" in name:
                bands["blue"] = file

            elif "b03" in name:
                bands["green"] = file

            elif "b04" in name:
                bands["red"] = file

        if len(bands) != 3:
            raise Exception(
                "Please upload B02, B03 and B04 GeoTIFF files."
            )

        with rasterio.open(bands["red"]) as rsrc, \
             rasterio.open(bands["green"]) as gsrc, \
             rasterio.open(bands["blue"]) as bsrc:

            r = normalize_band(rsrc.read(1))
            g = normalize_band(gsrc.read(1))
            b = normalize_band(bsrc.read(1))

            rgb = np.dstack([r, g, b])

            metadata = {
                "bands": 3,
                "width": rsrc.width,
                "height": rsrc.height,
                "crs": str(rsrc.crs)
            }

            return rgb, rsrc.transform, metadata

    # =====================================================
    # CASE 2 : Single GeoTIFF
    # =====================================================

    with rasterio.open(source) as src:

        transform = src.transform
        count = src.count

        metadata = {
            "bands": count,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs)
        }

        # -----------------------------
        # Single band
        # -----------------------------

        if count == 1:

            band = normalize_band(src.read(1))
            rgb = np.dstack([band, band, band])

        # -----------------------------
        # RGB
        # -----------------------------

        elif count == 3:

            r = normalize_band(src.read(1))
            g = normalize_band(src.read(2))
            b = normalize_band(src.read(3))

            rgb = np.dstack([r, g, b])

        # -----------------------------
        # More than 3 bands
        # -----------------------------

        else:

            r = normalize_band(src.read(1))
            g = normalize_band(src.read(2))
            b = normalize_band(src.read(3))

            rgb = np.dstack([r, g, b])

    return rgb, transform, metadata