import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost.sklearn import XGBClassifier, XGBRegressor


class XGBoostFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 10,
                     'max_depth': 3
                 }):

        if 'regression' == task:
            self.estimator = XGBRegressor(**params)
        else:
            self.estimator = XGBClassifier(**params)

    def fit(self, X, y, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit estimator and transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        self.model = self.estimator.fit(X, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        return self.one_hot_encoder_.fit_transform(self.model.apply(X))

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        return self.one_hot_encoder_.transform(self.model.apply(X))

    def predict_leafs(self, X, concate=True):
        gbdt_leaf = self.model.apply(X)
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        if concate:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class GBDTFeatureTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 10,
                     'max_depth': 3
                 }):

        if 'regression' == task:
            self.estimator = GradientBoostingRegressor(**params)
        else:
            self.estimator = GradientBoostingClassifier(**params)

    def fit(self, X, y, sample_weight=None):
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        self.model = self.estimator.fit(X, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        return self.one_hot_encoder_.fit_transform(
            self.model.apply(X)[:, :, 0])

    def transform(self, X):
        return self.one_hot_encoder_.transform(self.model.apply(X)[:, :, 0])

    def predict_leafs(self, X, concate=True):
        gbdt_leaf = self.model.apply(X)[:, :, 0]
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        if concate:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class LightGBMFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 10,
                     'max_depth': 3
                 }):

        if 'regression' == task:
            self.estimator = LGBMRegressor(**params)
        else:
            self.estimator = LGBMClassifier(**params)

    def fit(self, X, y, sample_weight=None):
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        self.model = self.estimator.fit(X, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        return self.one_hot_encoder_.fit_transform(
            self.model.predict(X, pred_leaf=True))

    def transform(self, X):
        return self.one_hot_encoder_.transform(
            self.model.predict(X, pred_leaf=True))

    def predict_leafs(self, X, concate=True):
        gbdt_leaf = self.model.predict(X, pred_leaf=True)
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        if concate:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class CatboostFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 10,
                     'max_depth': 3
                 }):

        if 'regression' == task:
            self.estimator = CatBoostRegressor(
                **params, logging_level='Silent', allow_writing_files=False)
        else:
            self.estimator = CatBoostClassifier(
                **params, logging_level='Silent', allow_writing_files=False)

    def fit(self, X, y, sample_weight=None):
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        self.model = self.estimator.fit(X, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        return self.one_hot_encoder_.fit_transform(
            self.model.calc_leaf_indexes(X))

    def transform(self, X):
        return self.one_hot_encoder_.transform(self.model.calc_leaf_indexes(X))

    def predict_leafs(self, X, concate=True):
        gbdt_leaf = self.model.calc_leaf_indexes(X)
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        if concate:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


if __name__ == '__main__':
    titanic = pd.read_csv('autotabular/datasets/data/Titanic.csv')
    # 'Embarked' is stored as letters, so fit a label encoder to the train set to use in the loop
    embarked_encoder = LabelEncoder()
    embarked_encoder.fit(titanic['Embarked'].fillna('Null'))
    # Record anyone travelling alone
    titanic['Alone'] = (titanic['SibSp'] == 0) & (titanic['Parch'] == 0)
    # Transform 'Embarked'
    titanic['Embarked'].fillna('Null', inplace=True)
    titanic['Embarked'] = embarked_encoder.transform(titanic['Embarked'])
    # Transform 'Sex'
    titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 0
    titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 1
    titanic['Sex'] = titanic['Sex'].astype('int8')
    # Drop features that seem unusable. Save passenger ids if test
    titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    trainMeans = titanic.groupby(['Pclass', 'Sex'])['Age'].mean()

    def f(x):
        if not np.isnan(x['Age']):  # not NaN
            return x['Age']
        return trainMeans[x['Pclass'], x['Sex']]

    titanic['Age'] = titanic.apply(f, axis=1)
    X_train = titanic.drop(['Survived'], axis=1)
    y_train = titanic['Survived']

    clf = XGBoostFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.transform(X_train)
    print(result)

    clf = LightGBMFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.transform(X_train)
    print(result)

    clf = GBDTFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.predict_leafs(X_train)
    print(result)

    clf = CatboostFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.predict_leafs(X_train)
    print(result)
