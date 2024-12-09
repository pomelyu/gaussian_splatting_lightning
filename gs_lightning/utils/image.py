import imageio as iio
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    return iio.v3.imread(image_path)
