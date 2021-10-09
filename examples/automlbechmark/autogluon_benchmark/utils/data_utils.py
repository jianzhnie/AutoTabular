from io import StringIO

import pandas as pd
from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.utils import infer_problem_type
from pandas import DataFrame, Series


def get_data_metadata(X: DataFrame, y: Series) -> dict:
    X_raw = convert_to_raw(X)

    feature_metadata_orig = FeatureMetadata.from_df(X)
    feature_metadata_raw = FeatureMetadata.from_df(X_raw)

    num_rows, num_cols = X.shape
    num_null = X.isnull().sum().sum()

    try:
        problem_type = infer_problem_type(y, silent=True)
    except TypeError:
        problem_type = infer_problem_type(y)
    if problem_type in ['binary', 'multiclass']:
        num_classes = len(y.unique())
    else:
        num_classes = None

    data_metadata = {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'num_null': num_null,
        'num_classes': num_classes,
        'problem_type': problem_type,
        'feature_metadata': feature_metadata_orig,
        'feature_metadata_raw': feature_metadata_raw,
    }
    # TODO: class imbalance
    # TODO: has_text
    # TODO: has_special
    # TODO: memory size

    return data_metadata


# Remove custom type information
def convert_to_raw(X, label=None):
    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = None
    with StringIO() as buffer:
        X.to_csv(buffer, index=True, header=True)
        buffer.seek(0)
        X = pd.read_csv(
            buffer, index_col=0, header=0, low_memory=False, encoding='utf-8')
    if label is not None:
        X[label] = y
    return X
