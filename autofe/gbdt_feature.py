import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm_reg_params = dict(
    objective='binary',
    subsample=0.8,
    min_child_weight=0.5,
    colsample_bytree=0.7,
    num_leaves=100,
    max_depth=12,
    learning_rate=0.05,
    n_estimators=10,
)


class GBDTFeatures(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 gbdt=None,
                 gbdt_params=None,
                 cv=CountVectorizer(
                     analyzer='word',
                     preprocessor=None,
                     ngram_range=(1, 1),
                     stop_words=None,
                     min_df=0,
                 )):
        self.gbdt = gbdt(**gbdt_params)
        self.cv = cv

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        return self

    def predict_proba(self, X):
        leaf = self.gbdt.predict(X, pred_leaf=True)
        leaf = (self.gbdt.predict(X, pred_leaf=True)).astype(str).tolist()
        # get the leaf
        gbdt_feats = self.gbdt.predict(X, pred_leaf=True)
        gbdt_feats_name = [
            'gbdt_leaf_' + str(i) for i in range(gbdt_feats.shape[1])
        ]
        # get the new datasets
        gbdt_feats_df = pd.DataFrame(gbdt_feats, columns=gbdt_feats_name)
        result = pd.concat([X, gbdt_feats_df], axis=1)
        if self.get_paramscv is not None:
            leaf = [' '.join(item) for item in leaf]
            result = self.cv.transform(leaf)
        return result
