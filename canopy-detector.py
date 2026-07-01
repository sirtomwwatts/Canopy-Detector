import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd


def calculate_canopy_cover(image, threshold="otsu"):
    """
    Calculate canopy cover from an image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image.
    threshold : int or "otsu"

    Returns
    -------
    canopy_percent
    sky_percent
    binary
    overlay
    used_threshold
    """

    # Convert RGB -> BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rgb = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    if threshold == "otsu":
        used_threshold, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        used_threshold = threshold
        _, binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

    # Calculate percentages
    sky_pixels = np.count_nonzero(binary == 255)
    canopy_pixels = np.count_nonzero(binary == 0)
    total_pixels = binary.size

    sky_percent = 100 * sky_pixels / total_pixels
    canopy_percent = 100 * canopy_pixels / total_pixels

    # Overlay
    overlay = rgb.copy()
    overlay[binary == 0] = [0, 255, 0]

    alpha = 0.4
    result = cv2.addWeighted(
        rgb,
        1 - alpha,
        overlay,
        alpha,
        0,
    )

    return (
        canopy_percent,
        sky_percent,
        binary,
        result,
        used_threshold,
    )


# -------------------------------------------------------
# Streamlit interface
# -------------------------------------------------------

st.set_page_config(
    page_title="Canopy Cover Calculator",
    layout="wide",
)

st.title("🌳 Canopy Cover Calculator")

st.write(
    "Upload an upward-looking canopy photograph to estimate "
    "canopy and sky cover."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    method = st.sidebar.radio(
        "Threshold Method",
        ["Otsu (Automatic)", "Manual"],
    )

    if method == "Manual":
        threshold = st.sidebar.slider("Threshold", 0, 255, 220)
    else:
        threshold = "otsu"

    canopy, sky, binary, overlay, used_threshold = calculate_canopy_cover(
        image,
        threshold
    )

    # -------------------------
    # VISUALS
    # -------------------------

    st.subheader("Visualisation")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", width="stretch")

    with col2:
        st.image(binary, caption="Binary Mask", width="stretch")

    st.image(overlay, caption="Canopy Overlay", width="stretch")

# -------------------------
# DOWNLOAD SECTION
# -------------------------

st.subheader("Download Results")

# CSV
csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📄 Download Results (CSV)",
    data=csv,
    file_name="canopy_results.csv",
    mime="text/csv",
)

# Overlay image download
overlay_pil = Image.fromarray(overlay)

overlay_buffer = io.BytesIO()
overlay_pil.save(overlay_buffer, format="PNG")

st.download_button(
    label="🌳 Download Overlay Image",
    data=overlay_buffer.getvalue(),
    file_name="canopy_overlay.png",
    mime="image/png",
)
