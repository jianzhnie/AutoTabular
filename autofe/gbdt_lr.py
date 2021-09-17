import warnings

import numpy as np
from scipy.sparse.construct import hstack
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
X, y = make_hastie_10_2(random_state=0)
x_train, x_test = X[:2000], X[2000:3000]
y_train, y_test = y[:2000], y[2000:3000]
gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
gb.fit(x_train, y_train)
score = roc_auc_score(y_test, gb.predict(x_test))
print('GBDT train data shape : {0}  auc: {1}'.format(x_train.shape, score))
lr = LogisticRegression()
lr.fit(x_train, y_train)
score = roc_auc_score(y_test, lr.predict(x_test))
print('LR train data shape : {0}  auc: {1}'.format(x_train.shape, score))
x_train_gb = gb.apply(x_train)[:, :, 0]
x_test_gb = gb.apply(x_test)[:, :, 0]
print(x_train_gb.shape)
gb_onehot = OneHotEncoder()
x_trains = gb_onehot.fit_transform(
    np.concatenate((x_train_gb, x_test_gb), axis=0))
print(x_trains.shape)
rows = x_train.shape[0]
lr = LogisticRegression()
x_train_gb_data = x_trains[:rows, :]
x_test_gb_data = x_trains[rows:, :]
lr.fit(x_train_gb_data, y_train)
score = roc_auc_score(y_test, lr.predict(x_test_gb_data))
print('LR with GBDT apply data, train data shape : {0}  auc: {1}'.format(
    x_train_gb_data.shape, score))
lr = LogisticRegression()
x_train_merge = hstack([x_trains[:rows, :], x_train])
x_test_merge = hstack([x_trains[rows:, :], x_test])
lr.fit(x_train_merge, y_train)
score = roc_auc_score(y_test, lr.predict(x_test_merge))
print(
    'LR with GBDT apply data and origin data, train data shape : {0}  auc: {1}'
    .format(x_train_merge.shape, score))
