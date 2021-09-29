import numpy as np
import pandas as pd
from autofe.feature_engineering.gbdt_feature import CatboostFeatureTransformer, GBDTFeatureTransformer, LightGBMFeatureTransformer, XGBoostFeatureTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    rows = titanic.shape[0]
    n_train = int(rows * 0.77)
    train_data = titanic[:n_train, :]
    test_data = titanic[n_train:, :]

    X_train = titanic.drop(['Survived'], axis=1)
    y_train = titanic['Survived']

    clf = XGBoostFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.concate_transform(X_train)
    print(result)

    clf = LightGBMFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.concate_transform(X_train)
    print(result)

    clf = GBDTFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.concate_transform(X_train)
    print(result)

    clf = CatboostFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    result = clf.concate_transform(X_train)
    print(result)

    lr = LogisticRegression()
    x_train_gb, x_test_gb, y_train_gb, y_test_gb = train_test_split(
        result, y_train)
    x_train, x_test, y_train, y_test = train_test_split(X_train, y_train)

    lr.fit(x_train, y_train)
    score = roc_auc_score(y_test, lr.predict(x_test))
    print('LR with GBDT apply data, train data shape : {0}  auc: {1}'.format(
        x_train.shape, score))

    lr = LogisticRegression()
    lr.fit(x_train_gb, y_train_gb)
    score = roc_auc_score(y_test_gb, lr.predict(x_test_gb))
    print('LR with GBDT apply data, train data shape : {0}  auc: {1}'.format(
        x_train_gb.shape, score))
