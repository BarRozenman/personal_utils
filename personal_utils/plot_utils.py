import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def scatter_data_in_2d(features: np.ndarray, labels: np.ndarray=None, dim_reduction_method='pca'):
    if features.ndim ==1 or features.shape[1] ==1:
        feat =np.asarray([np.arange(features.shape[0]),np.asarray(features).flatten()]).T
    elif features.shape[1] > 2:
        if dim_reduction_method == 'pca':
            feat = PCA(2).fit_transform(features)
        else:
            feat = TSNE(3).fit_transform(features)
    else:
        feat = copy.deepcopy(features)
    if labels is None:
        labels = np.zeros(len(features))
    elif not isinstance(labels,np.ndarray):
        labels =np.asarray(labels)
    plt.figure()
    for i in set(labels):
        plt.scatter(*feat[labels == i, :].T)
    plt.show()

def scatter_data_in_3d(features: np.ndarray, labels: np.ndarray, dim_reduction_method='pca'):
    if features.shape[1] > 3:
        if dim_reduction_method == 'pca':
            feat = PCA(3).fit_transform(features)
        else:
            feat = TSNE(3).fit_transform(features)
    else:
        feat = copy.deepcopy(features)
    plt.figure()
    for i in set(labels):
        plt.scatter(*feat[labels == i, :].T)
