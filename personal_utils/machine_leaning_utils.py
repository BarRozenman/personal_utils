import pandas as pd
from featurewiz import featurewiz


def feature_selection(data_and_lbls:pd.DataFrame,target_name):
    features, train = featurewiz(data_and_lbls, target_name, corr_limit=0.7, verbose=2, sep=",",
    header=0,test_data="", feature_engg="", category_encoders="")
    return features