import PIL
from PIL import Image
from typing import Union, Tuple

import cv2
import numpy as np
import torch


def xywh2xyxy(x):
    boxes = x.clone()
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    return boxes


def plot_one_number(centroid, img, color=None, label=None):
    # color = color or [np.random.randint(0, 255) for _ in range(3)]
    p1, p2 = centroid
    # cv2.rectangle(img, p1, p2, color, 2, lineType=cv2.LINE_AA)

    if label is None:
        label = "obj"
    # t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
    # p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
    # cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (p1, p2),
        0,
        fontScale=2,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return img


def plot_one_box(box, img, color=None, label=None):
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, 2, lineType=cv2.LINE_AA)

    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2),
            0,
            0.5,
            [255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def scale_boxes(boxes, orig_shape, new_shape):
    if boxes.ndim == 1:
        boxes_ = torch.from_numpy(boxes[:, np.newaxis])
    else:
        boxes_ = torch.from_numpy(boxes)

    H, W = orig_shape
    nH, nW = new_shape
    gain = min(nH / H, nW / W)
    pad = (nH - H * gain) / 2, (nW - W * gain) / 2

    boxes_[:, ::2] -= pad[1]
    boxes_[:, 1::2] -= pad[0]
    boxes_[:, :4] /= gain

    boxes_[:, ::2].clamp_(0, orig_shape[1])
    boxes_[:, 1::2].clamp_(0, orig_shape[0])
    return boxes_.round().numpy()


def bbox_center_of_mass(box: np.ndarray, format="lurl") -> Tuple[float, float]:
    """
      Crops an object in an image using [left,upper,right,lower] (lurl) bounding box or [top,left,bottom,right] (tlbr)

    Inputs:
      box: one box from Detectron2 pred_boxes
    """
    if format == "tlbr":
        top = box[0]
        left = box[1]
        bottom = box[2]
        right = box[3]
        x_center = (right + left) / 2
        y_center = (bottom + top) / 2

    elif format == "lurl":

        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        x_center = (x_top_left + x_bottom_right) / 2
        y_center = (y_top_left + y_bottom_right) / 2
    else:
        print("non valid bbox format")
        return None, None
    return x_center, y_center


def crop_object_bbox(
    image: Union[PIL.Image.Image, np.ndarray], box: np.ndarray
) -> Union[PIL.Image.Image, np.ndarray]:
    """
      Crops an object in an image using lurl [left,upper,right,lower] bounding box

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

    crop_img = image.crop(
        (int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right))
    )
    if isinstance(image, np.ndarray):
        crop_img = np.asarray(crop_img)
    return crop_img
