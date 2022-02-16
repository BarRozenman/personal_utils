from typing import Union

import PIL
import numpy as np
from PIL import Image


def crop_object(image: Union[PIL.Image.Image, np.ndarray], box: np.ndarray) -> Union[PIL.Image.Image, np.ndarray]:
    """
    Crops an object in an image using [left,upper,right,lower] bounding box

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
    if isinstance(image, np.ndarray):
        crop_img = np.asarray(crop_img)
    return crop_img
