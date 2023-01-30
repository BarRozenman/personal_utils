import logging
from functools import partial

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def vector2integer_encoding(vector_labels: np.ndarray) -> np.ndarray:
    """
    transform vector encoding (for example one hot encoding to integer/categorical encoding
    (one_hot_encoding)
    >>> vector2integer_encoding(np.asarray([[1,0,0],[0,0,1]]))
    array([0, 2])
    >>> vector2integer_encoding(np.asarray([[1,2,0],[0.3,0.4,0.8]]))
    array([1, 2])
    """
    if vector_labels.ndim == 1:
        logging.warning("vector_labels is already in a integer_encoding format...")
        return vector_labels

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(np.arange(vector_labels.shape[1]).reshape(-1, 1))
    integer_labels = (
        one_hot_encoder.inverse_transform(vector_labels).flatten().astype(int)
    )
    return integer_labels


def integer2vector_encoding(
    integer_labels: np.ndarray, categories_num=None
) -> np.ndarray:
    """
    transform integer/categorical encoding to vector encoding (for example one hot encoding)

    >>> integer2vector_encoding(np.asarray([0,2]))
    array([[1., 0., 0.],
           [0., 0., 1.]])
    >>> integer2vector_encoding(np.asarray([1, 2]))
    array([[0., 1., 0.],
           [0., 0., 1.]])

    """
    try:
        integer_labels = np.asarray(integer_labels)
    except Exception as e:
        ValueError(
            "integer2vector_encoding() accepts only array-like argument as input"
        )

    integer_labels = np.asarray(
        integer_labels
    )  # in case we get a pd.Series or other type
    if isinstance(integer_labels[0], str):
        le = LabelEncoder()
        le.fit(integer_labels)
        integer_labels = le.transform(integer_labels)
    deduced_categories_num = np.max(
        (len(set(integer_labels)), np.max(integer_labels) + 1)
    )
    if not categories_num:
        categories_num = deduced_categories_num
    elif categories_num < deduced_categories_num:
        logging.error(
            f'argument "categories_num"-{categories_num} is smaller than the '
            f'"deduced_categories_num"-{deduced_categories_num}'
        )
        return
    categories_array = np.arange(categories_num).reshape(-1, 1)  # existing categories
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(categories_array)
    vector_labels = one_hot_encoder.transform(integer_labels[:, np.newaxis]).toarray()
    return vector_labels


def vector2string_encoding(
    vector_labels: np.ndarray, catg_names, suppress_warn=False
) -> np.ndarray:
    le = LabelEncoder()
    le.fit(catg_names)

    if isinstance(vector_labels, list):
        vector_labels = np.array(vector_labels)

    if vector_labels.ndim == 1:
        if not suppress_warn:
            logging.warning(
                '"vector_labels" is in a "integer_encoding" format,'
                ' use function "integer2string_encoding"'
            )
        integer_labels = np.asarray(vector_labels).astype(int)
    else:
        integer_labels = vector2integer_encoding(vector_labels)
    string_labels = le.inverse_transform(integer_labels)
    return string_labels


def is_one_hot_encoding(array: np.ndarray) -> bool:
    if array.squeeze().ndim == 1:
        return False
    is_one_hot_encoding = (
        (array.sum(axis=1) - np.ones(array.shape[0])).sum() == 0
        if len(array) > 0
        else None
    )
    return is_one_hot_encoding


def is_integer_encoding(array: np.ndarray) -> bool:
    try:
        array = np.asarray(array)
    except Exception as e:
        raise ValueError(
            f"is_integer_encoding() accepts only array-like argument as input, {e}"
        )
    if array.squeeze().ndim != 1:
        return False
    if array.dtype.char == "U":
        return False
    is_integer_encoding = (array == array.astype(int)).all()
    return is_integer_encoding


def is_vector_encoding(array: np.ndarray) -> bool:
    """
    check if array is vector encoding, meaning one hot encoding or any other soft labeling option
    Args:
        array:

    Returns:

    Examples
    --------
    >>> array = np.asarray([[2.3,1,3,4.5],[2.3,1.6,6,4.5]])
    >>> is_vector_encoding(array)
    True

    >>> array = np.asarray([[2,1,3,4],[2,1,6,4]])
    >>> is_vector_encoding(array)
    True

    >>> array = np.asarray([[2,1,3,4]]])
    >>> is_vector_encoding(array)
    True

    >>> array = np.asarray([[0,1,0,0],[0,0,0,1]])
    >>> is_vector_encoding(array)
    True

    >>> array = np.asarray([[0,'b','a',0],[0,0,0,1]])
    >>> is_vector_encoding(array)
    False

    >>> array = np.asarray([0,0,0,1])
    >>> is_vector_encoding(array)
    False # since it has 1 dimention
    """
    if is_one_hot_encoding(array):
        return True
    elif array.ndim == 2 and (array.dtype == np.int or array.dtype == np.float):
        return True
    else:
        return False


integer2string_encoding = partial(vector2string_encoding, suppress_warn=True)

string2vector_encoding = partial(integer2vector_encoding)

# string2integer_encoding = partial(vector2integer_encoding)
