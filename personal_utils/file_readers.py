from pathlib import Path
from typing import Union

import cv2
import filetype
import numpy as np
from PIL import Image


def read_image(file_path: Union[str, Path]) -> np.ndarray:
    file_path = str(file_path)
    img = Image.open(file_path)
    return np.array(img)


def read_video(file_path: Union[str, Path])->cv2.VideoCapture:
    file_path = str(file_path)
    file = cv2.VideoCapture(file_path)
    return file


def read_audio(file_path: Union[str, Path])->np.ndarray:
    pass


def read_file(file_path: Union[str, Path])->np.ndarray:
    """can rea any file type image sound or video"""
    file_path = str(file_path)
    func_dict = {"image": read_image, "video": read_video, "audio": read_audio}
    file_type = filetype.guess(file_path).mime.split("/")[0]
    file_content = func_dict.get(file_type)(file_path)
    return file_content
