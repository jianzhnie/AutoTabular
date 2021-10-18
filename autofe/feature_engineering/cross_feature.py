from typing import List, Tuple

import pandas as pd
from autofe.feature_engineering.groupby import get_candidate_categorical_feature
from pandas.core.frame import DataFrame
from sklearn.exceptions import NotFittedError


# This class does not represent any sctructural advantage, but I keep it to
# keep things tidy, as guidance for contribution and because is useful for the
# check_is_fitted function
class BaseTransformer:
    """Base Class of All Preprocessors."""

    def __init__(self, *args):
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')


def check_is_fitted(
    estimator: BaseTransformer,
    attributes: List[str] = None,
    all_or_any: str = 'all',
    condition: bool = True,
):
    r"""Checks if an estimator is fitted

    Parameters
    ----------
    estimator: ``BasePreprocessor``,
        An object of type ``BasePreprocessor``
    attributes: List, default = None
        List of strings with the attributes to check for
    all_or_any: str, default = "all"
        whether all or any of the attributes in the list must be present
    condition: bool, default = True,
        If not attribute list is passed, this condition that must be True for
        the estimator to be considered as fitted
    """

    estimator_name: str = estimator.__class__.__name__
    error_msg = (
        "This {} instance is not fitted yet. Call 'fit' with appropriate "
        'arguments before using this estimator.'.format(estimator_name))
    if attributes is not None and all_or_any == 'all':
        if not all([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif attributes is not None and all_or_any == 'any':
        if not any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif not condition:
        raise NotFittedError(error_msg)


class CrossFeatureTransformer(BaseTransformer):

    def __init__(self,
                 category_cols: List[str],
                 continue_cols: List[str],
                 crossed_cols: List[Tuple[str, str]] = None):
        super(CrossFeatureTransformer, self).__init__()

        self.category_cols = category_cols
        self.crossed_cols = crossed_cols
        self.continue_cols = continue_cols

    def fit(self, df: pd.DataFrame) -> BaseTransformer:
        df_cat = self._prepare_cat(df)
        self.col_crossed_cols = df_cat.columns.tolist()

    def _cross_cols(self, df: pd.DataFrame):
        df_cc = df.copy()
        crossed_colnames = []
        for cols in self.crossed_cols:
            for c in cols:
                df_cc[c] = df_cc[c].astype('str')
            colname = '_'.join(cols)
            df_cc[colname] = df_cc[list(cols)].apply(
                lambda x: '-'.join(x), axis=1)

            crossed_colnames.append(colname)
        return df_cc[crossed_colnames]

    def _prepare_cat(self, df: pd.DataFrame):
        if self.category_cols is None:
            self.category_cols = get_candidate_categorical_feature(
                df, self.target_name, threshold=0.9)
        if self.crossed_cols is not None:
            df_cc = self._cross_cols(df)
            return pd.concat([df[self.category_cols], df_cc], axis=1)
        else:
            return df.copy()[self.category_cols]


class CrossColTransform():

    def __init__(self,
                 df: DataFrame,
                 category_cols: List[str] = None,
                 continue_cols: List[str] = None,
                 crossed_cols: List[Tuple[str, str]] = None,
                 target_name: str = None):

        self.df = df
        self.category_cols = category_cols
        self.crossed_cols = crossed_cols
        self.continue_cols = continue_cols
        self.target_name = target_name

        if self.category_cols is None:
            self.category_cols = self.get_category_columns(df, target_name)
        if self.continue_cols is None:
            self.continue_cols = self.get_numerical_columns(df, target_name)
        if self.crossed_cols is None:
            self.crossed_cols = self.get_cross_columns(self.category_cols)

        self.df, self.crossed_colnames = self.generate_cross_cols(
            self.crossed_cols)
        self.category_cols.extend(self.crossed_colnames)

    def get_cross_columns(self, category_cols):
        crossed_cols = []
        for i in range(0, len(category_cols) - 1):
            for j in range(i + 1, len(category_cols)):
                crossed_cols.append((category_cols[i], category_cols[j]))
        return crossed_cols

    def get_category_columns(self, df, target):
        cat_col_names = []
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] and col != target:
                cat_col_names.append(col)
        return cat_col_names

    def get_numerical_columns(self, df, target):
        num_col_names = []
        for col in df.columns:
            if df[col].dtype in ['float', 'int'] and col != target:
                num_col_names.append(col)
        return num_col_names

    def generate_cross_cols(self, crossed_cols):
        df_cc = self.df.copy()
        crossed_colnames = []
        for cols in crossed_cols:
            for c in cols:
                df_cc[c] = df_cc[c].astype('str')
            colname = '_'.join(cols)
            df_cc[colname] = df_cc[list(cols)].apply(
                lambda x: '-'.join(x), axis=1)

            crossed_colnames.append(colname)
        return df_cc, crossed_colnames

    def generate_groupby_feature(self, methods, reserve=False):
        df = self.df.copy()
        new_col_names = []
        for cat_col in self.category_cols:
            for num_col in self.continue_cols:
                for method in methods:
                    new_col_name = cat_col + '_' + num_col + '_' + method
                    new_col_names.append(new_col_name)
                    df[new_col_name] = df.groupby(cat_col)[num_col].transform(
                        method)
                    df = df.fillna(0)
        if reserve:
            return df
        else:
            return df[new_col_names]


if __name__ == '__main__':
    titanic = pd.read_csv('autotabular/datasets/data/Titanic.csv')
    print(titanic)
    # cf = CrossFeatureTransformer(
    #     category_cols=['Pclass', 'Sex', 'Embarked'],
    #     continue_cols=['Age', 'Fare'],
    #     crossed_cols=[('Sex', 'Embarked'), ('Pclass', 'Sex')])

    # df = cf._prepare_cat(titanic)
    # print(df)
    # print(cf._make_column_feature_list(df))

    cf = CrossColTransform(df=titanic, target_name='Survived')
    df = cf.generate_groupby_feature(methods=['mean'])
    print(df)
