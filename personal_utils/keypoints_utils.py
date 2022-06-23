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
    # plt.plot(d["left_eye"] [1]- d["left_eye"], marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
    # plt.plot(d["right_eye"][1], d["right_eye"][0], marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
    delta_y = int(np.linalg.norm(d["left_ear"] - d["right_ear"]) * 2)
    delta_x = int(np.linalg.norm(d["left_ear"] - d["right_ear"]) * 2.7)
    cropped_image = image[y - delta_y : y + delta_y, x - delta_x : x + delta_x]
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


if __name__ == "__main__":
    keypoints = [
        [412.5, 913.5555419921875, 0.7784045934677124],
        [402.65625, 920.111083984375, 0.8434509038925171],
        [402.65625, 910.2777709960938, 0.815990149974823],
        [399.375, 929.9444580078125, 0.8828603029251099],
        [389.53125, 903.7222290039062, 0.8851001858711243],
        [432.1875, 936.5, 0.8991664052009583],
        [419.0625, 877.5, 0.8581922054290771],
        [484.6875, 956.1666870117188, 0.8600435256958008],
        [471.5625, 851.2777709960938, 0.9124369621276855],
        [540.46875, 969.2777709960938, 0.8925716876983643],
        [520.78125, 851.2777709960938, 0.8957035541534424],
        [530.625, 920.111083984375, 0.7871749401092529],
        [524.0625, 877.5, 0.8002240061759949],
        [609.375, 913.5555419921875, 0.7120108604431152],
        [602.8125, 877.5, 0.8405110836029053],
        [668.4375, 907.0, 0.4874367117881775],
        [658.59375, 861.111083984375, 0.45792704820632935],
    ]
    keypoints = [
        [395.75, 1083.8333740234375, 0.7226657867431641],
        [392.0208435058594, 1087.5555419921875, 0.8011261224746704],
        [392.0208435058594, 1083.8333740234375, 0.6100142598152161],
        [399.4791564941406, 1087.5555419921875, 0.8208721280097961],
        [395.75, 1109.888916015625, 0.8706275820732117],
        [425.5833435058594, 1076.388916015625, 0.8526957035064697],
        [425.5833435058594, 1132.22216796875, 0.9293864369392395],
        [477.7916564941406, 1057.77783203125, 0.8586650490760803],
        [477.7916564941406, 1147.111083984375, 0.8777715563774109],
        [515.0833129882812, 1039.1666259765625, 0.7603127360343933],
        [522.5416870117188, 1158.27783203125, 0.8749631643295288],
        [530.0, 1091.27783203125, 0.8121324777603149],
        [530.0, 1128.5, 0.8533005118370056],
        [604.5833129882812, 1091.27783203125, 0.887996256351471],
        [600.8541870117188, 1121.0555419921875, 0.8865742683410645],
        [690.3541870117188, 1095.0, 0.7697999477386475],
        [686.625, 1117.3333740234375, 0.8695598840713501],
    ]
    image = np.array(
        PIL.Image.open(
            "/home/graphecy/projects/track_project/simple-object-tracking/temp.png"
        )
    )
    get_head_from_keypoints(keypoints, image)
