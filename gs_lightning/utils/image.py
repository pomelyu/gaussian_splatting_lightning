import imageio as iio
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    return iio.v3.imread(image_path)

def save_image(image: np.ndarray, image_path: str) -> None:
    Image.fromarray(image).save(image_path)
