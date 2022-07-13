import struct
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


def read_video(file_path: Union[str, Path]) -> cv2.VideoCapture:
    file_path = str(file_path)
    file = cv2.VideoCapture(file_path)
    return file


def read_audio(file_path: Union[str, Path]) -> np.ndarray:
    pass


def read_file(file_path: Union[str, Path]) -> np.ndarray:
    """can rea any file type image sound or video"""
    file_path = str(file_path)
    func_dict = {"image": read_image, "video": read_video, "audio": read_audio}
    file_type = filetype.guess(file_path).mime.split("/")[0]
    file_content = func_dict.get(file_type)(file_path)
    return file_content


def read_pfm(file_path: str) -> np.ndarray:
    """ Read a PFM (Portable FloatMap) file, and return a numpy array 

    Args:
        file_path (str): path to .pfm file

    Returns:
        np.ndarray: float image
    """
    with Path(file_path).open("rb") as pfm_file:

        line1, line2, line3 = (
            pfm_file.readline().decode("latin-1").strip() for _ in range(3)
        )
        assert line1 in ("PF", "Pf")

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale
