from typing import Dict, Tuple

import cv2
import numpy as np
import seaborn as sns


def closest_color(list_of_colors, color):
    # Find the closest color to the detected one based on the predefined palette
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance
# Color Detection with K-means
def detect_color(img: np.ndarray, cv_mask:np.ndarray, palette: Dict[str,Tuple]=None) -> Tuple:
    if palette is None:
        palette = palette_basic
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.reshape((img.shape[1] * img.shape[0], 3))
    img = img[cv_mask == 1, :]
    np.sum(img.mean(0)-np.asarray(list(palette.values())))
    # kmeans = KMeans(n_clusters=2)
    # s = kmeans.fit(img)
    #
    # labels = kmeans.labels_
    # centroid = kmeans.cluster_centers_
    # labels = list(labels)
    # percent = []
    #
    # for i in range(len(centroid)):
    #     j = labels.count(i)
    #     j = j / (len(labels))
    #     percent.append(j)
    #
    # detected_color = centroid[np.argmin(percent)]
    # list_of_colors = list(palette.values())
    # assigned_color = closest_color(list_of_colors, detected_color)[0]
    # assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))
    #
    # if assigned_color == (0, 0, 0):
    #     assigned_color = (128, 128, 128)

    return assigned_color

def get_masked_mean_color(img: np.ndarray, cv_mask:np.ndarray) -> Tuple:
    """

    :param img: BRG! image that will be converted to RGB
    :param cv_mask:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.reshape((img.shape[1] * img.shape[0], 3))
    img = img[cv_mask == 1, :]

    return img.mean(0)

palette_100 = {count: [int(255 * j) for j in i] for count, i in enumerate(sns.color_palette("Spectral", 100))}
palette_basic = {'b': (0, 0, 128),
           'g': (0, 128, 0),
           'r': (255, 0, 0),
           'c': (0, 192, 192),
           'm': (192, 0, 192),
           'y': (192, 192, 0),
           'k': (0, 0, 0),
           'w': (255, 255, 255)}