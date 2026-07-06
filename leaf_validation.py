import numpy as np
import cv2
from PIL import Image


def validate_leaf(image):
    """
    Validate whether the uploaded image is likely
    to contain a green leaf.

    Returns
    -------
    is_leaf : bool
    message : str
    """

    # -------------------------------
    # Convert PIL Image to NumPy
    # -------------------------------

    img = np.array(image)

    if img is None:
        return False, "Unable to read image."

    # -------------------------------
    # Resize
    # -------------------------------

    img = cv2.resize(img, (224, 224))

    # -------------------------------
    # HSV conversion
    # -------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    # -------------------------------
    # Green Mask
    # -------------------------------

    lower_green = np.array([25, 40, 40])

    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    green_ratio = np.sum(mask > 0) / mask.size

    # -------------------------------
    # Edge Detection
    # -------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        80,
        180
    )

    edge_ratio = np.sum(edges > 0) / edges.size

    # -------------------------------
    # Texture
    # -------------------------------

    texture = np.std(gray)

    # -------------------------------
    # Decision Rules
    # -------------------------------

    if green_ratio < 0.08:

        return False, "Image contains very little green vegetation."

    if texture < 15:

        return False, "Image texture is not consistent with a leaf."

    if edge_ratio < 0.02:

        return False, "Leaf structure not detected."

    return True, "Leaf detected."