import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
