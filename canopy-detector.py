import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_canopy_cover(image_path, threshold="otsu", show_plots=True):
    """
    Calculate canopy cover from an upward-looking forest image.

    Parameters
    ----------
    image_path : str
        Path to the local image.
    threshold : int or "otsu", optional
        Threshold value (0-255) or "otsu" for automatic thresholding.
    show_plots : bool, optional
        Whether to display the images.

    Returns
    -------
    canopy_percent : float
    sky_percent : float
    binary : ndarray
    """

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert for plotting
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    if isinstance(threshold, str) and threshold.lower() == "otsu":
        used_threshold, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        used_threshold = threshold
        _, binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

    # Count pixels
    sky_pixels = np.count_nonzero(binary == 255)
    canopy_pixels = np.count_nonzero(binary == 0)
    total_pixels = binary.size

    sky_percent = 100 * sky_pixels / total_pixels
    canopy_percent = 100 * canopy_pixels / total_pixels

    print(f"Threshold used: {used_threshold:.1f}")
    print(f"Canopy: {canopy_percent:.2f}%")
    print(f"Sky: {sky_percent:.2f}%")

    # Create overlay
    overlay = rgb.copy()
    overlay[binary == 0] = [0, 255, 0]

    alpha = 0.4
    result = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)

    # Display
    if show_plots:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap="gray")
        plt.title(f"Threshold = {used_threshold:.1f}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title("Canopy Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return canopy_percent, sky_percent, binary
