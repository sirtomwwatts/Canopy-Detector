import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Canopy Cover Analyzer",
    layout="wide"
)


# =====================================================
# CORE PROCESSING (cached for speed)
# =====================================================
@st.cache_data
def calculate_canopy_cover(image_array, threshold="otsu"):
    """
    image_array: RGB numpy array
    """

    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    rgb = image_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    if threshold == "otsu":
        used_threshold, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        used_threshold, binary = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY
        )

    sky_pixels = np.count_nonzero(binary == 255)
    canopy_pixels = np.count_nonzero(binary == 0)
    total_pixels = binary.size

    sky_percent = 100 * sky_pixels / total_pixels
    canopy_percent = 100 * canopy_pixels / total_pixels

    # Overlay
    overlay = rgb.copy()
    overlay[binary == 0] = [0, 255, 0]

    result = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    return canopy_percent, sky_percent, binary, overlay, result, used_threshold


# =====================================================
# UI HEADER
# =====================================================
st.title("🌳 Canopy Cover Analyzer")
st.write("Upload a canopy image to estimate sky vs vegetation cover.")


# =====================================================
# SIDEBAR CONTROLS (clean separation)
# =====================================================
st.sidebar.header("Settings")

method = st.sidebar.radio(
    "Threshold method",
    ["Otsu (Automatic)", "Manual"],
    key="threshold_method"
)

threshold = None
if method == "Manual":
    threshold = st.sidebar.slider(
        "Threshold",
        0, 255, 220,
        key="threshold_slider"
    )
else:
    threshold = "otsu"


# =====================================================
# FILE UPLOAD (ONLY ONE — FIXES YOUR ERROR)
# =====================================================
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    key="uploader"
)


# =====================================================
# MAIN APP LOGIC
# =====================================================
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # Run analysis
    canopy, sky, binary, overlay, result, used_threshold = calculate_canopy_cover(
        image,
        threshold
    )

    # =================================================
    # METRICS
    # =================================================
    col1, col2, col3 = st.columns(3)

    col1.metric("🌳 Canopy", f"{canopy:.2f}%")
    col2.metric("☁️ Sky", f"{sky:.2f}%")
    col3.metric("Threshold", f"{used_threshold:.1f}")


    # =================================================
    # VISUALISATION
    # =================================================
    st.subheader("Visualisation")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", width="stretch")

    with col2:
        st.image(binary, caption="Binary Mask", width="stretch")

    st.image(result, caption="Canopy Overlay", width="stretch")


    # =================================================
    # DOWNLOADS
    # =================================================
    st.subheader("Downloads")

    # CSV
    results_df = pd.DataFrame({
        "Metric": ["Canopy (%)", "Sky (%)", "Threshold"],
        "Value": [
            round(canopy, 2),
            round(sky, 2),
            round(float(used_threshold), 1),
        ],
    })

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download CSV",
        csv,
        "canopy_results.csv",
        "text/csv"
    )

    # Overlay image
    overlay_buffer = io.BytesIO()
    Image.fromarray(result).save(overlay_buffer, format="PNG")

    st.download_button(
        "🌳 Download Overlay",
        overlay_buffer.getvalue(),
        "canopy_overlay.png",
        "image/png"
    )

else:
    st.info("Upload an image to begin analysis.")
