import copy
import itertools
import logging
from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import machine_learning_utils as ml_utils

""" this line (_ = Axes3D) exists so then we reformat the file we do not delete this import since python doesn't not recognize we use it but we do """
_ = Axes3D


def scatter_image(
    x: float, y: float, image: Union[str, Path, np.ndarray], ax=None, zoom: float = 0.5
):
    """show the image at the given coordinates"""
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def scatter_multiple_images(
    x: np.ndarray,
    y: np.ndarray,
    images: List[Union[str, Path, np.ndarray]],
    ax=None,
    zoom: float = 0.5,
    shown_amount='max'
):
    if shown_amount == "max":
        shown_amount = len(images)
        rand_indexes = range(shown_amount)
    else:
        rand_indexes = np.round(
            np.random.uniform(low=0, high=len(images), size=(shown_amount,))
        ).astype(int)
    """show the image at the gi ven coordinates"""
    artists = []
    for idx,(curr_x, curr_y, im) in enumerate(zip(x, y, images)):
        if im is None:
            continue
        if idx not in rand_indexes:
            continue
        art = scatter_image(curr_x, curr_y, im, ax=ax, zoom=zoom)
        artists.append(art)
    return artists


def scatter_data_in_2d(
    features: np.ndarray,
    labels: np.ndarray = None,
    dim_reduction_method="pca",
    show_3d_as_size=True,
):
    if features.ndim == 1 or features.shape[1] == 1:
        feat = np.asarray(
            [np.arange(features.shape[0]), np.asarray(features).flatten()]
        ).T

    elif features.shape[1] > 2:
        if dim_reduction_method == "pca":
            feat = PCA(3).fit_transform(features)
        else:
            feat = TSNE(3).fit_transform(features)
    else:
        feat = copy.deepcopy(features)
    if labels is None:
        labels = np.zeros(len(features))
    elif not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    plt.figure()
    for i in set(labels):
        plt.scatter(*feat[labels == i, :].T)


def scatter_clustering_with_gt_labels_in_3d(  # to do refactor
    x: np.ndarray,
    cluster_labels: np.ndarray,
    gt_y=None,
    labels_names=None,
    title=None,
    dim_reduction_method="pca",
    shown_amount=30,
):  # x_embedded, y_true):
    """using pca to show 3 dims  scatter in 3d
    x.shape = [samples_num,features_num]

    Examples
    --------
    see Examples in scatter_clustering_with_gt_labels_in_2d, it has the same API
    """
    logger = logging.getLogger(__name__)
    if cluster_labels is None:
        logger.warning(
            f'argument "cluster_labels" was no set, calculating clusterd labels independently'
        )
        clustering_params, _ = ml_utils.get_best_clustering_algo_params(x)
        cluster_labels = ml_utils.apply_clustering_algo(clustering_params, x)

    if shown_amount == "max":
        shown_amount = len(cluster_labels)
    rand_indexes = np.round(
        np.random.uniform(low=0, high=len(cluster_labels), size=(shown_amount,))
    ).astype(int)
    if isinstance(gt_y, (list, np.ndarray)) and not all(
        isinstance(float(e).is_integer(), (int, float)) for e in gt_y
    ):
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit(gt_y)
        gt_y = lb.transform(gt_y).flatten()
        labels_names = lb.classes_.tolist()
    if x.shape[1] != 2:
        if dim_reduction_method == "pca":
            x = ml_utils.pca_dim_reduction(x, 3)
        elif dim_reduction_method == "manifold":
            x = ml_utils.manifold_dim_reduction(x, 3)
    if labels_names is None:
        labels_names = range(len(np.unique(cluster_labels)))
    if isinstance(cluster_labels, list):
        cluster_labels = np.array(cluster_labels)
    markers = ["o", ",", "x", "+", "v", "^", "<", ">", "s", "d", "."][
        : len(labels_names)
    ]
    colors = cm.rainbow(np.linspace(0, 1, len(labels_names)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if gt_y is None:
        for i, t in enumerate(set(cluster_labels)):
            idx = cluster_labels == t
            ax.scatter(
                x[idx, 0][:shown_amount],
                x[idx, 1][:shown_amount],
                x[idx, 2][:shown_amount],
                label=t,
                s=180,
                c=colors[i].reshape(1, 4),
                marker=markers[i],
                alpha=0.5,
            )
    else:
        for idx in range(x.shape[0]):
            try:
                ax.text(
                    x[idx, 0],
                    x[idx, 1],
                    x[idx, 2],
                    labels_names[gt_y[idx]],
                    color=plt.cm.nipy_spectral(cluster_labels[idx] / 10.0),
                    fontdict={"weight": "bold", "size": 9},
                    alpha=0.5,
                )
            except Exception as e:
                logger.error(f"could not write text on figure - {e}")
        vmax = np.max(x)
        vmin = np.min(x)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_zlim(vmin, vmax)
    plt.legend()
    if isinstance(title, str):
        fig.suptitle(title)
    plt.title(f"number of samples:{x.shape[0]} \n  number of features:{x.shape[1]}")


def scatter_clustering_with_gt_labels_in_2d(
    x: np.ndarray,
    cluster_labels: np.ndarray = None,
    gt_y=None,
    gt_labels_names=None,
    title: str = None,
    dim_reduction_method="pca",
    shown_amount="max",
):
    """
    using pca to show 2 dims  scatter in 2d
    the color is the cluster labels and the text is the gt labels

    data.shape = [num_of_sample,num_of_features]

    Examples
    --------
    df = sns.load_dataset('iris')
    features = df.iloc[:,:-1]
    labels = pd.factorize(df['species'])[0]
    params, score = ml_utils.get_best_clustering_algo_params(features, cluster_num_list=[3])
    cluster_labels = ml_utils.apply_clustering_algo(params, features)
    scatter_clustering_with_gt_labels_in_2d(features,cluster_labels,labels)


    """
    logger = logging.getLogger(__name__)
    if cluster_labels is None:
        logger.warning(
            f'argument "cluster_labels" was no set, calculating clusters labels independently'
        )
        clustering_params, _ = ml_utils.get_best_clustering_algo_params(x)
        cluster_labels = ml_utils.apply_clustering_algo(clustering_params, x)

    if isinstance(cluster_labels, list):
        cluster_labels = np.array(cluster_labels).flatten()

    if shown_amount == "max":
        shown_amount = len(cluster_labels)
        rand_indexes = range(shown_amount)
    else:
        rand_indexes = np.round(
            np.random.uniform(low=0, high=len(cluster_labels), size=(shown_amount,))
        ).astype(int)
    logging.info(
        f"we assume number of samples:{x.shape[0]} \n  number of features:{x.shape[1]}"
    )
    if x.shape[1] != 2:
        if dim_reduction_method == "pca":
            x = ml_utils.pca_dim_reduction(x, 2)
        elif dim_reduction_method == "manifold":
            x = ml_utils.manifold_dim_reduction(x, 2)
        elif dim_reduction_method == "tsne":
            x = ml_utils.tsne_dim_reduction(x, 2)

    if cluster_labels is None:
        cluster_labels = range(x.shape[0])

    if isinstance(gt_y, (list, np.ndarray)) and not all(
        isinstance(float(e).is_integer(), (int, float)) for e in gt_y
    ):
        # todo use vector2integer encoding func here
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit(gt_y)
        gt_y = lb.transform(gt_y).flatten()
        gt_labels_names = lb.classes_.tolist()
    if gt_labels_names is None:
        gt_labels_names = range(len(np.unique(gt_y)))

    # x_min, x_max = np.min(x), np.max(x)
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    X_red = (x - (x_min)) / (x_max - (x_min)) - 0.1
    X_red = X_red
    markers = itertools.cycle(["o", ",", "x", "+", "v", "^", "<", ">", "s", "d", "."])

    fig, ax = plt.subplots()
    colors = cm.jet(np.linspace(0, 1, len(np.unique(cluster_labels))))
    np.random.shuffle(colors)
    cluster_labels_dict = dict(zip(np.unique(cluster_labels), range(len(cluster_labels))))
    gt_labels_dict = dict(zip(np.unique(gt_y), gt_labels_names))
    if gt_y is None:
        for t, color, marker in zip(set(cluster_labels), colors, markers):
            idx = cluster_labels == t
            ax.scatter(
                x[idx, 0][:shown_amount],
                x[idx, 1][:shown_amount],
                label=cluster_labels_dict[t],
                s=120,
                c=color.reshape(1, 4),
                marker=marker,
                alpha=0.5,
            )
        # plt.scatter(X_red[:, 0], X_red[:, 1])
    else:
        aaa= []
        for i in range(X_red.shape[0]):
            if i not in rand_indexes:
                continue
            try:
                ax.text(
                    X_red[i, 0],
                    X_red[i, 1],
                    gt_labels_dict[gt_y[i]],
                    color=colors[cluster_labels_dict[cluster_labels[i]]],
                    fontdict={"weight": "bold", "size": 9},
                )
            except Exception as e:
                logger.error(f"could not plot sting on figure -  {e}")

    plt.xticks([])
    plt.yticks([])
    if isinstance(title, str):
        fig.suptitle(title)
    plt.title(f"number of samples:{x.shape[0]} \n  number of features:{x.shape[1]}")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# def scatter_data_in_3d(features: np.ndarray, labels: np.ndarray, dim_reduction_method='pca'):
#     if features.shape[1] > 3:
#         if dim_reduction_method == 'pca':
#             feat = PCA(3).fit_transform(features)
#         else:
#             feat = TSNE(3).fit_transform(features)
#     else:
#         feat = copy.deepcopy(features)
#     plt.figure()
#     for i in set(labels):
#         plt.scatter(*feat[labels == i, :].T)
