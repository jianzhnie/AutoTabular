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
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/bank_marketing/'

    RESULTS_DIR = ROOTDIR / 'results/bank_marketing/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    bank_maket = pd.read_csv(PROCESSED_DATA_DIR / 'bankm.csv')
    target_name = 'target'

    IndList = range(bank_maket.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = bank_maket.iloc[train_list]
    val = bank_maket.iloc[val_list]
    test = bank_maket.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, eval_metric='roc_auc', path=RESULTS_DIR).fit(
            train_data=train, tuning_data=val)

    scores = predictor.evaluate(test, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(test)

    # gbdt transformer
    cat_col_names, cont_col_names = [], []
    for col in bank_maket.columns:
        if bank_maket[col].dtype == 'O' and col != 'target':
            cat_col_names.append(col)
        elif bank_maket[col].dtype == 'float' and col != 'target':
            cont_col_names.append(col)

    num_classes = len(set(bank_maket[target_name].values.ravel()))
    label_encoder = LabelEncoder(cat_col_names)
    bank_maket = label_encoder.fit_transform(bank_maket)

    X = bank_maket.drop(target_name, axis=1)
    y = bank_maket[target_name]
    print(X.columns)
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
