import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def feature_selection(data_and_lbls: pd.DataFrame, target_name):
    from featurewiz import featurewiz

    features, train = featurewiz(
        data_and_lbls,
        target_name,
        corr_limit=0.7,
        verbose=2,
        sep=",",
        header=0,
        test_data="",
        feature_engg="",
        category_encoders="",
    )
    return features


def reduce_dims(x, y, return_model=False):  # do to refatcor
    from skpp import ProjectionPursuitRegressor

    # plot pca components
    # import scikitplot as skplt
    # pca = PCA(random_state=1)
    # pca.fit(x)
    # skplt.decomposition.plot_pca_component_variance(pca)

    estimator = ProjectionPursuitRegressor(r=13)
    ans = estimator.fit_transform(x, y)
    # estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))
    # from . import plot_utils
    # plot_utils.scatter_clustering_with_gt_labels_in_2d(ans,y)
    if return_model:
        return pd.DataFrame(ans), estimator

    return pd.DataFrame(ans)
    import hnswlib
    import numpy as np
    import pickle

    """ like knn but much faster (380 times faster)"""

    dim = 128
    num_elements = 10000

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))
    ids = np.arange(num_elements)

    # Declaring index
    p = hnswlib.Index(space="l2", dim=dim)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Element insertion (can be called several times):
    p.add_items(data, ids)

    # Controlling the recall by setting ef:
    p.set_ef(50)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=1)

    # Index objects support pickling
    # WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
    # Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
    p_copy = pickle.loads(
        pickle.dumps(p)
    )  # creates a copy of index p using pickle round-trip

    ### Index parameters are exposed as class properties:
    print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
    print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    print(
        f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}"
    )
    print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")


def pca_dim_reduction(mat, num_dims=5):
    num_dims = np.min([mat.shape[0], num_dims])
    pca = PCA(n_components=num_dims, random_state=42)
    PCA_mat = pca.fit_transform(mat)
    return PCA_mat


def tsne_dim_reduction(mat, num_dims=5):
    num_dims = np.min([mat.shape[0], num_dims])
    tsne = TSNE(n_components=num_dims)
    tsne_results = tsne.fit_transform(mat)
    return tsne_results
