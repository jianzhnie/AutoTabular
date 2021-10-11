import os
from pathlib import Path

import pandas as pd
from autofe.tabular_embedding.tabular_embedding_transformer import TabularEmbeddingTransformer
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

    train = pd.read_csv(PROCESSED_DATA_DIR / 'adult_train.csv')
    valid = pd.read_csv(PROCESSED_DATA_DIR / 'adult_val.csv')
    test = pd.read_csv(PROCESSED_DATA_DIR / 'adult_test.csv')

    target_name = 'target'
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
        # 'num_stack_levels': 1,
        # 'num_bag_sets': 1,
        'verbosity': 2,
    }
    print(train.head())

    cat_col_names = []
    for col in train.columns:
        if train[col].dtype == 'object' and col != 'target':
            cat_col_names.append(col)

    num_col_names = []
    for col in train.columns:
        if train[col].dtype == 'float' and col != 'target':
            num_col_names.append(col)

    num_classes = len(set(train[target_name].values.ravel()))

    print(num_classes)
    print(cat_col_names)
    print(num_col_names)

    X_train = train.drop([target_name], axis=1)
    y_train = train[target_name]

    X_val = valid.drop([target_name], axis=1)
    y_val = valid[target_name]

    X_test = test.drop([target_name], axis=1)
    y_test = test[target_name]

    predictor = run_autogluon(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label=target_name,
        init_args=init_args,
        fit_args=fit_args)

    scores = predictor.evaluate(test, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test)

    # GBDT embeddings
    transformer = TabularEmbeddingTransformer(
        cat_col_names=cat_col_names,
        num_col_names=num_col_names,
        date_col_names=[],
        target_name=target_name,
        num_classes=num_classes)

    print(transformer)
    train_transform = transformer.fit_transform(train)
    val_trainform = transformer.transform(valid)
    test_transform = transformer.transform(test)

    X_train_enc = train_transform.drop([target_name], axis=1)
    X_val_enc = val_trainform.drop([target_name], axis=1)
    X_test_enc = test_transform.drop([target_name], axis=1)

    predictor = run_autogluon(
        X_train=X_train_enc,
        y_train=y_train,
        X_val=X_val_enc,
        y_val=y_val,
        label=target_name,
        init_args=init_args,
        fit_args=fit_args)

    X_test_enc[target_name] = y_test
    scores = predictor.evaluate(X_test)
    leaderboard = predictor.leaderboard(X_test_enc)
    print(leaderboard)
