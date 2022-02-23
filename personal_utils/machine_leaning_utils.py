import pandas as pd
from featurewiz import featurewiz
from skpp import ProjectionPursuitRegressor


def feature_selection(data_and_lbls:pd.DataFrame,target_name):
    features, train = featurewiz(data_and_lbls, target_name, corr_limit=0.7, verbose=2, sep=",",
    header=0,test_data="", feature_engg="", category_encoders="")
    return features

def reduce_dims(x,y):
    estimator = ProjectionPursuitRegressor()
    estimator.fit_transform(x,y)
    # estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))