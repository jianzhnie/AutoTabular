"""Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection using XGBoost. We optimize both the choice of booster model and its hyperparameters.
"""

import optuna
import pandas as pd
from autofe.get_feature import get_baseline_total_data
from sklearn.metrics import r2_score
from xgboost.sklearn import XGBRegressor


def objective(trial):
    param = {
        'verbosity': 0,
        # sampling ratio for training data.
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        # sampling according to each tree.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        # learning_rate
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
    }

    clf = XGBRegressor(**param).fit(X_train, y_train)
    pred = clf.predict(X_test)
    r2 = r2_score(y_test, pred)
    return r2


if __name__ == '__main__':
    root_path = './data/processed_data/house/'
    train_data = pd.read_csv(root_path + 'train_data.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'SalePrice'
    """lr baseline"""
    total_data_base = get_baseline_total_data(total_data)
    train_data = total_data_base.iloc[:len_train]
    test_data = total_data_base.iloc[len_train:]
    # total_data_base = total_data_base.sample(frac=0.3)
    # train_data, test_data = train_test_split(total_data_base, test_size=0.3)

    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=6000)

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

    clf = XGBRegressor(**param).fit(X_train, y_train)
    preds = clf.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(r2)
