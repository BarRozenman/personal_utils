import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .flags import flags


def get_head_from_keypoints(keypoints, image):
    """
    Returns the head numpy array image from an image and coco keypoints.
    "keypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
     "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
     "right_knee", "left_ankle", "right_ankle" ]
    """
    # Get the head from the keypoints
    a = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    head = np.array(keypoints)[:, :2]
    d = dict(zip(a, head))
    delta = max(np.abs(d["right_eye"] - d["left_eye"]))
    top = d["nose"][0]
    top = (d["nose"] + np.abs(d["right_ear"] - d["left_ear"]))[0]
    left = (d["left_ear"])[1]
    left = (d["left_ear"] - np.abs(d["right_ear"] - d["left_ear"]))[1]
    height = np.abs(d["right_ear"] - d["left_ear"])[1]
    width = np.abs(d["right_ear"] - d["left_ear"])[1]
    top, left, height, width = map(int, (top, left, height, width))
    im_size = image.shape
    x = int(d["nose"][1])
    y = int(d["nose"][0])
    delta_y = int(np.linalg.norm(d["left_ear"] - d["right_ear"]) * 2)
    delta_x = int(np.linalg.norm(d["left_ear"] - d["right_ear"]) * 2.7)
    cropped_image = image[y - delta_y : y + delta_y, x - delta_x : x + delta_x]
    cropped_image = np.flip(cropped_image, 0)
    if flags.verbose:
        """plot keypoints to make sure the are in the proper location"""
        plt.imshow(image)
        for i in keypoints:
            plt.plot(
                i[1],
                i[0],
                marker="o",
                markersize=1,
                markeredgecolor="red",
                markerfacecolor="green",
            )
        plt.show()
        plt.figure()
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.show()
    return cropped_image
