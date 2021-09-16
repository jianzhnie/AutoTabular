import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost.sklearn import XGBClassifier, XGBRegressor


class BaseGbdtEmbedding(BaseEstimator):

    def __init__(self, base_estimatior=None):
        self.base_estimator = base_estimatior

    def fit(self, X, y=None, sample_weight=None):
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
        self.base_estimator.fit(X, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
        return self.one_hot_encoder_.fit_transform(self.apply(X))

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
        return self.one_hot_encoder_.transform(self.apply(X))


def get_estimator_class(task, estimator_name):
    """when adding a new learner, need to add an elif branch."""

    if 'xgboost' == estimator_name:
        if 'regression' == task:
            estimator_class = XGBRegressor
        else:
            estimator_class = XGBClassifier
    elif 'rf' == estimator_name:
        if 'regression' == task:
            estimator_class = RandomForestRegressor
        else:
            estimator_class = RandomForestClassifier
    elif 'gbdt' == estimator_name:
        if 'regression' == task:
            estimator_class = GradientBoostingClassifier
        else:
            estimator_class = GradientBoostingRegressor
    elif 'lgbm' == estimator_name:
        if 'regression' == task:
            estimator_class = LGBMRegressor
        else:
            estimator_class = LGBMClassifier
    elif 'catboost' == estimator_name:
        if 'regression' == task:
            estimator_class = CatBoostClassifier
        else:
            estimator_class = CatBoostRegressor
    else:
        raise ValueError(
            estimator_name + ' is not a built-in learner. '
            'Please use AutoML.add_learner() to add a customized learner.')
    return estimator_class


class GBDTFeaturesCountVecTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 gbdt=None,
                 gbdt_params=None,
                 vectorizer=CountVectorizer(
                     analyzer='word',
                     preprocessor=None,
                     ngram_range=(1, 1),
                     stop_words=None,
                     min_df=0,
                 )):
        self.gbdt = gbdt(**gbdt_params)
        self.vectorizer = vectorizer

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        leaf = (self.gbdt.predict(X, pred_leaf=True)).astype(str).tolist()
        leaf = [' '.join(item) for item in leaf]
        self.result = self.vectorizer.fit_transform(leaf)
        return self

    def predict_proba(self, X):
        leaf = self.gbdt.predict(X, pred_leaf=True)
        leaf = (self.gbdt.predict(X, pred_leaf=True)).astype(str).tolist()
        if self.vectorizer is not None:
            leaf = [' '.join(item) for item in leaf]
            result = self.vectorizer.transform(leaf)
        return result


class XGBoostFeatureTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self, task, concate_fea=False):

        self.task = task
        self.concate_fea = concate_fea

        self.xgb = XGBClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
        self.onehot = OneHotEncoder()

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        return self

    def predict_proba(self, X):
        gbdt_leaf = self.gbdt.apply(X)[:, :, 0]
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        result = pd.concat([X, gbdt_feats], axis=1)
        return result


class GBDTFeatureTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self) -> None:
        super().__init__()

        self.gbdt = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
        self.onehot = OneHotEncoder()

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        return self

    def predict_proba(self, X):
        gbdt_leaf = self.gbdt.apply(X)[:, :, 0]
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        result = pd.concat([X, gbdt_feats], axis=1)
        return result


class LightGBMFeatureTransformer(BaseEstimator, ClassifierMixin):

    def __init__(self, gbdt=None, gbdt_params=None):
        self.gbdt = gbdt(**gbdt_params)

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        return self

    def predict_proba(self, X):
        gbdt_leaf = self.gbdt.predict(X, pred_leaf=True)
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_leaf.shape[1])
        ]
        gbdt_feats = pd.DataFrame(gbdt_leaf, columns=gbdt_feats_name)
        result = pd.concat([X, gbdt_feats], axis=1)
        return result


if __name__ == '__main__':
    import lightgbm as lgb
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

    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 64,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 4,
        'verbose': 0
    }
    clf = GBDTFeaturesCountVecTransformer(
        gbdt=lgb.LGBMClassifier, gbdt_params=params)
    clf.fit(X_train, y_train)
    result = clf.predict_proba(X_train)
    print(result)

    clf = LightGBMFeatureTransformer(
        gbdt=lgb.LGBMClassifier, gbdt_params=params)
    clf.fit(X_train, y_train)
    result = clf.predict_proba(X_train)
    print(result)

    clf = GBDTFeatureTransformer()
    clf.fit(X_train, y_train)
    result = clf.predict_proba(X_train)
    print(result)
