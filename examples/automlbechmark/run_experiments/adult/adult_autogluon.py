import os
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from autogluon.tabular import TabularPredictor
from pytorch_widedeep.utils import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult_autogluon.csv')
    target_name = 'income'

    IndList = range(adult_data.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = adult_data.iloc[train_list]
    val = adult_data.iloc[val_list]
    test = adult_data.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, path=RESULTS_DIR).fit(
            train_data=train, tuning_data=val)

    scores = predictor.evaluate(test, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(test)

    # gbdt transformer
    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'
    cat_col_names, cont_col_names = [], []
    for col in adult_data.columns:
        if adult_data[col].dtype == 'O' or col != 'target':
            cat_col_names.append(col)
        elif col != 'target':
            cont_col_names.append(col)

    num_classes = len(set(adult_data[target_name].values.ravel()))
    label_encoder = LabelEncoder(cat_col_names)
    adult_data = label_encoder.fit_transform(adult_data)

    X = adult_data.drop(target_name, axis=1)
    y = adult_data[target_name]

    # GBDT embeddings
    clf = LightGBMFeatureTransformer(
        task='classification', categorical_feature=cat_col_names)
    clf.fit(X, y)
    X_enc = clf.concate_transform(X, concate=False)
    selector = SelectFromModel(
        estimator=LogisticRegression(), max_features=64).fit(X_enc, y)
    support = selector.get_support()
    col_names = X_enc.columns[support]
    X_enc = selector.transform(X_enc)
    X_enc = pd.DataFrame(X_enc, columns=col_names)

    X_enc = pd.concat([X, X_enc])
    X_enc[target_name] = y
    train_enc = X_enc.iloc[train_list]
    val_enc = X_enc.iloc[val_list]
    test_enc = X_enc.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, path=RESULTS_DIR).fit(
            train_data=train_enc, tuning_data=val_enc)

    scores = predictor.evaluate(test_enc, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(test_enc)
