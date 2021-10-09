import os
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from autogluon.tabular import TabularPredictor

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    MODELS_DIR = ROOTDIR / 'results/adult/models/autogluon'
    if not MODELS_DIR.is_dir():
        os.makedirs(MODELS_DIR)

    train = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_train.pkl')
    valid = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_val.pkl')
    test = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_test.pkl')

    target = 'target'
    metric = 'accuracy'
    predictor = TabularPredictor(
        label=target, eval_metric=metric, path=MODELS_DIR).fit(
            train_data=train, tuning_data=valid)
    predictor.leaderboard(test)
    perf = predictor.evaluate(test, auxiliary_metrics=False)
    print(perf)

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    clf = LightGBMFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)
    X_train_enc = clf.concate_transform(X_train)
    X_val_enc = clf.concate_transform()

    predictor = TabularPredictor(
        label=target, eval_metric=metric, path=MODELS_DIR).fit(
            train_data=X_train_enc, tuning_data=valid)
    predictor.leaderboard(test)
    perf = predictor.evaluate(test, auxiliary_metrics=False)
    print(perf)
