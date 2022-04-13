from pathlib import Path
from typing import Union

import cv2
import filetype
import numpy as np


def read_image(filepath:  Union[str, Path]):
    filepath = str(filepath)
    try:
        original_img = cv2.imread(filepath, cv2.IMREAD_COLOR)  # load original image

        rgb_original_img = cv2.cvtColor(
            original_img.astype(np.uint8), cv2.COLOR_BGR2RGB
        )
    except:
        raise Exception("could not read image, duplicate names or broke")

    return rgb_original_img


def read_video(filepath: Union[str, Path]):
    filepath = str(filepath)
    file = cv2.VideoCapture(filepath)
    return file


def read_audio(filepath: Union[str, Path]):
    filepath = str(filepath)
    pass


def read_file(filepath: Union[str, Path]):
    """can rea any file type image sound or video"""
    filepath = str(filepath)
    func_table = {"image": read_image, "video": read_video, "audio": read_audio}
    file_type = filetype.guess(filepath).mime.split("/")[0]
    file = func_table.get(file_type)(filepath)
    return file
