import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter

from scripts.detector import detect_objects
from scripts.draw import draw_detections
from scripts.validator import validate_image
from scripts.geotiff import read_geotiff
from scripts.coordinate import attach_coordinates

st.set_page_config(
    page_title="Airport Infrastructure Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Airport & Airstrip Infrastructure Detection System")
st.divider()

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("Input")

uploaded_files = st.sidebar.file_uploader(
    "Upload Image / GeoTIFF",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True
)

st.sidebar.info(
"""
Supported Inputs

• JPG
• PNG
• JPEG

• Single-band GeoTIFF

• RGB GeoTIFF

• Sentinel-2 Bands
    - B02
    - B03
    - B04
"""
)

# ======================================================
# Main
# ======================================================

if uploaded_files:

    transform = None
    crs = None
    is_geotiff = False

    # --------------------------------------------------
    # Decide input type
    # --------------------------------------------------

    first_name = uploaded_files[0].name.lower()

    if first_name.endswith((".tif", ".tiff")):

        # -----------------------------
        # One GeoTIFF
        # -----------------------------

        if len(uploaded_files) == 1:

            rgb, transform, metadata = read_geotiff(
                uploaded_files[0]
            )

        # -----------------------------
        # Sentinel Bands
        # -----------------------------

        elif len(uploaded_files) == 3:

            rgb, transform, metadata = read_geotiff(
                uploaded_files
            )

        else:

            st.error(
                "Please upload either:\n\n"
                "• One GeoTIFF\n\n"
                "OR\n\n"
                "• B02 + B03 + B04 Sentinel bands."
            )

            st.stop()

        is_geotiff = True
        crs = metadata["crs"]

    else:

        if len(uploaded_files) > 1:

            st.error(
                "For JPG / PNG upload only ONE image."
            )

            st.stop()

        rgb = np.array(
            Image.open(uploaded_files[0]).convert("RGB")
        )

    # ======================================================
    # Validation
    # ======================================================

    status, checks = validate_image(rgb)

    st.sidebar.subheader("Image Validation")

    for icon, msg in checks:
        st.sidebar.write(f"{icon} {msg}")

    if status == "red":

        st.error(
            "Image cannot be processed.\n\n"
            "Please upload a valid airport image."
        )

        st.stop()

    elif status == "yellow":

        st.warning(
            "Image quality is not ideal.\n"
            "Detection accuracy may reduce."
        )

    else:

        st.sidebar.success("Ready for Detection")
            # ======================================================
    # Detection
    # ======================================================

    _, detections = detect_objects(rgb)

    if transform is not None:

        detections = attach_coordinates(
            detections,
            transform
        )

    annotated = draw_detections(
        rgb,
        detections
    )

    # ======================================================
    # Images
    # ======================================================

    left, right = st.columns(2)

    with left:

        st.subheader("Original Image")

        st.image(
            rgb,
            use_container_width=True
        )

    with right:

        st.subheader("Detection Result")

        st.image(
            annotated,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # Detection Summary
    # ======================================================

    st.subheader("Detection Summary")

    counts = Counter(
        [d["class"] for d in detections]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total",
        len(detections)
    )

    c2.metric(
        "Runways",
        counts.get("runway", 0)
    )

    c3.metric(
        "Hangars",
        counts.get("hangar", 0)
    )

    c4.metric(
        "Terminal Buildings",
        counts.get("terminal building", 0)
    )

    st.divider()

    # ======================================================
    # Objects
    # ======================================================

    st.subheader("Detected Infrastructure")

    if len(detections) == 0:

        st.warning(
            "No airport infrastructure detected."
        )

    else:

        runway = 1
        hangar = 1
        terminal = 1

        for d in detections:

            cls = d["class"]
            conf = d["confidence"]

            if cls == "runway":

                title = f"🛣️ Runway {runway}"
                runway += 1

            elif cls == "hangar":

                title = f"🏢 Hangar {hangar}"
                hangar += 1

            elif cls == "terminal building":

                title = f"🏬 Terminal Building {terminal}"
                terminal += 1

            else:

                title = cls

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(f"### {title}")

                with col2:

                    st.metric(
                        "Confidence",
                        f"{conf*100:.1f}%"
                    )

                st.markdown(
                    f"### 📍 Pixel Center: ({d['center_x']}, {d['center_y']})"
                )

                if "priority_level" in d:

                    st.write(
                        f"**Priority:** {d['priority_level']}"
                    )

                if "priority" in d:

                    st.write(
                        f"**Priority Score:** {d['priority']}"
                    )

                if "latlon" in d:

                    lat, lon = d["latlon"]

                    st.write(
                        f"**Latitude:** {lat}"
                    )

                    st.write(
                        f"**Longitude:** {lon}"
                    )
                        # ======================================================
    # GeoTIFF Information
    # ======================================================

    if is_geotiff:

        st.divider()

        st.subheader("GeoTIFF Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Bands",
                metadata["bands"]
            )

        with c2:
            st.metric(
                "Width",
                metadata["width"]
            )

        with c3:
            st.metric(
                "Height",
                metadata["height"]
            )

        st.markdown("### Coordinate Reference System")

        st.code(crs)

# ======================================================
# No Upload
# ======================================================

else:

    st.info(
        """
Upload an airport image to begin.

Supported:

• JPG

• PNG

• Single-band GeoTIFF

• RGB GeoTIFF

• Sentinel-2 (B02 + B03 + B04)
"""
    )