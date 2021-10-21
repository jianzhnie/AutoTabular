"""Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection using XGBoost. We optimize both the choice of booster model and its hyperparameters.
"""

import numpy as np
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
from autofe.get_feature import get_baseline_total_data
from sklearn.metrics import roc_auc_score


def objective(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    param = {
        'verbosity':
        0,
        'objective':
        'binary:logistic',
        # defines booster, gblinear for linear functions.
        'booster':
        trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        # L2 regularization weight.
        'lambda':
        trial.suggest_float('lambda', 1e-8, 10, log=True),
        # L1 regularization weight.
        'alpha':
        trial.suggest_float('alpha', 1e-8, 10, log=True),
        # sampling ratio for training data.
        'subsample':
        trial.suggest_float('subsample', 0.3, 1.0),
        # sampling according to each tree.
        'colsample_bytree':
        trial.suggest_float('colsample_bytree', 0.3, 1.0),
        # learning_rate
        'learning_rate':
        trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth':
        trial.suggest_int('max_depth', 2, 12, step=2),
        'n_estimators':
        trial.suggest_int('n_estimators', 100, 1000, step=200),
    }

    if param['booster'] in ['gbtree', 'dart']:
        # maximum depth of the tree, signifies complexity of the tree.
        param['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param['min_child_weight'] = trial.suggest_int('min_child_weight', 2,
                                                      10)
        param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        param['grow_policy'] = trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide'])

    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical(
            'sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical(
            'normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float(
            'rate_drop', 1e-8, 1.0, log=True)
        param['skip_drop'] = trial.suggest_float(
            'skip_drop', 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy


if __name__ == '__main__':
    root_path = './data/processed_data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'target'
    """lr baseline"""
    total_data_base = get_baseline_total_data(total_data)

    train_data = total_data_base.iloc[:len_train]
    test_data = total_data_base.iloc[len_train:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200, timeout=6000)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print(trial.params)
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    train_data = total_data_base.iloc[:len_train]
    test_data = total_data_base.iloc[len_train:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    param = trial.params
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    acc = sklearn.metrics.accuracy_score(y_test, pred_labels)
    auc = roc_auc_score(y_test, preds)
    print(acc, auc)
