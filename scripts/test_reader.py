from scripts.geotiff import GeoTIFFReader
from PIL import Image

reader = GeoTIFFReader()

rgb, metadata = reader.load_rgb(
    "test/input/B04_(Raw).tiff",
    "test/input/B03_(Raw).tiff",
    "test/input/B02_(Raw).tiff"
)

print(metadata)

Image.fromarray(rgb).save("outputs/test_rgb.png")

print("RGB image saved successfully!")
