import os
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from autogluon.tabular import TabularPredictor


def run_autogluon(X_train,
                  y_train,
                  X_val,
                  y_val,
                  label: str,
                  init_args: dict = None,
                  fit_args: dict = None):
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}

    X_train[label] = y_train
    X_val[label] = y_val
    predictor = TabularPredictor(
        label=label, **init_args).fit(
            train_data=X_train, tuning_data=X_val, **fit_args)

    return predictor


if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    train = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_train.pkl')
    valid = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_val.pkl')
    test = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_test.pkl')

    target = 'target'

    init_args = {'eval_metric': 'roc_auc', 'path': RESULTS_DIR}
    fit_args = {
        'time_limit': 1500,
        # 'use_bag_holdout': True,
        'hyperparameters': {
            'KNN': {},
            'RF': {},
            'GBM': {},
        },
        # 'num_bag_folds': 5,
        'num_stack_levels': 1,
        'num_bag_sets': 1,
        'verbosity': 2,
    }

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_val = valid.drop([target], axis=1)
    y_val = valid[target]

    X_test = test.drop([target], axis=1)
    y_test = test[target]

    predictor = run_autogluon(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label=target,
        init_args=init_args,
        fit_args=fit_args)

    scores = predictor.evaluate(test, auxiliary_metrics=False)
    print(scores)
    leaderboard = predictor.leaderboard(test)
    print(leaderboard)

    clf = LightGBMFeatureTransformer(task='classification')
    clf.fit(X_train, y_train)

    X_train_enc = clf.concate_transform(X_train)
    X_val_enc = clf.concate_transform(X_val)
    X_test_enc = clf.concate_transform(X_test)

    predictor = run_autogluon(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label=target,
        init_args=init_args,
        fit_args=fit_args)

    X_test_enc[target] = y_test
    scores = predictor.evaluate(X_test)
    leaderboard = predictor.leaderboard(X_test_enc)
    print(leaderboard)
