import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

IMG_SIZE = 224

def lbp_rgb(image):
    lbp_channels = []

    for c in range(3):
        lbp = local_binary_pattern(
            image[:, :, c],
            P=8,
            R=1,
            method="uniform"
        )

        lbp = lbp / (lbp.max() + 1e-8)
        lbp_channels.append(lbp)

    return np.stack(lbp_channels, axis=-1)


def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    rgb = np.array(image).astype(np.float32)

    rgb_preprocessed = preprocess_input(rgb)

    lbp = lbp_rgb(rgb.astype(np.uint8))

    rgb_preprocessed = np.expand_dims(rgb_preprocessed, axis=0)
    lbp = np.expand_dims(lbp, axis=0)

    return image, rgb_preprocessed, lbp