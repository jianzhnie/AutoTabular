"""Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection using XGBoost. We optimize both the choice of booster model and its hyperparameters.
"""

import optuna
import pandas as pd
from autofe.get_feature import get_baseline_total_data
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier


def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        # sampling ratio for training data.
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        # sampling according to each tree.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        # learning_rate
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 12, step=2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
    }

    clf = XGBClassifier(**param).fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_prob)
    return auc


if __name__ == '__main__':
    root_path = './data/processed_data/credit/'
    train_data = pd.read_csv(root_path + 'train_data.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'payment'
    """lr baseline"""
    total_data_base = get_baseline_total_data(total_data)
    total_data_base = total_data_base.sample(frac=0.3)
    train_data, test_data = train_test_split(total_data_base, test_size=0.3)

    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print(trial.params)
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    param = trial.params

    total_data_base = get_baseline_total_data(total_data)
    train_data = total_data_base.iloc[:len_train]
    test_data = total_data_base.iloc[len_train:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    clf = XGBClassifier(**param).fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_prob)
    print(acc, auc)
