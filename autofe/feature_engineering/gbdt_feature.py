import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier, XGBRegressor


class XGBoostFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 100,
                     'max_depth': 3
                 }):
        self.short_name = 'xbgoost'
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

    def dense_transform(self, X, keep_original=True):
        gbdt_leaf = self.model.apply(X)
        onehot_embedding = self.one_hot_encoder_.transform(gbdt_leaf).toarray()
        gbdt_feats_name = [
            f'{self.short_name}' + '_embed_' + str(i)
            for i in range(onehot_embedding.shape[1])
        ]
        gbdt_feats = pd.DataFrame(onehot_embedding, columns=gbdt_feats_name)
        if keep_original:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class GBDTFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 100,
                     'max_depth': 3
                 }):
        self.short_name = 'GBDT'
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

    def dense_transform(self, X, keep_original=True):
        gbdt_leaf = self.model.apply(X)[:, :, 0]
        onehot_embedding = self.one_hot_encoder_.transform(gbdt_leaf).toarray()
        gbdt_feats_name = [
            f'{self.short_name}' + '_embed_' + str(i)
            for i in range(onehot_embedding.shape[1])
        ]
        gbdt_feats = pd.DataFrame(onehot_embedding, columns=gbdt_feats_name)
        if keep_original:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class LightGBMFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 categorical_feature='auto',
                 params={
                     'n_estimators': 100,
                     'max_depth': 3
                 }):
        self.short_name = 'lightgbm'
        if 'regression' == task:
            self.estimator = LGBMRegressor(**params)
        else:
            self.estimator = LGBMClassifier(**params)

        self.categorical_feature = categorical_feature

    def fit(self, X, y):
        self.fit_transform(X, y, categorical_feature=self.categorical_feature)
        return self

    def fit_transform(self, X, y=None, categorical_feature='auto'):
        self.model = self.estimator.fit(X, y)
        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        return self.one_hot_encoder_.fit_transform(
            self.model.predict(X, pred_leaf=True))

    def transform(self, X):
        return self.one_hot_encoder_.transform(
            self.model.predict(X, pred_leaf=True))

    def dense_transform(self, X, keep_original=True):
        gbdt_leaf = self.model.predict(X, pred_leaf=True)
        onehot_embedding = self.one_hot_encoder_.transform(gbdt_leaf).toarray()
        gbdt_feats_name = [
            f'{self.short_name}' + '_embed_' + str(i)
            for i in range(onehot_embedding.shape[1])
        ]
        gbdt_feats = pd.DataFrame(onehot_embedding, columns=gbdt_feats_name)
        if keep_original:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats


class CatboostFeatureTransformer(BaseEstimator):

    def __init__(self,
                 task='regression',
                 params={
                     'n_estimators': 100,
                     'max_depth': 3
                 }):

        self.short_name = 'catboost'

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

    def dense_transform(self, X, keep_original=True):
        gbdt_leaf = self.model.calc_leaf_indexes(X)
        onehot_embedding = self.one_hot_encoder_.transform(gbdt_leaf).toarray()
        gbdt_feats_name = [
            f'{self.short_name}' + '_embed_' + str(i)
            for i in range(onehot_embedding.shape[1])
        ]
        gbdt_feats = pd.DataFrame(onehot_embedding, columns=gbdt_feats_name)
        if keep_original:
            return pd.concat([X, gbdt_feats], axis=1)
        else:
            return gbdt_feats
