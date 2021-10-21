"""Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection using XGBoost. We optimize both the choice of booster model and its hyperparameters.
"""

import optuna
import pandas as pd
from autofe.deeptabular_utils import LabelEncoder
from autofe.get_feature import get_baseline_total_data, get_category_columns
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier


def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        # sampling ratio for training data.
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        # sampling according to each tree.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        # learning_rate
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
    }

    clf = XGBClassifier(**param).fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc


if __name__ == '__main__':
    root_path = './data/processed_data/shelter/'
    train_data = pd.read_csv(root_path + 'train_data.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'OutcomeType'
    """lr baseline"""
    cat_col_names = get_category_columns(total_data, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data_base = label_encoder.fit_transform(total_data)

    train_data = total_data_base.iloc[:len_train]
    test_data = total_data_base.iloc[len_train:]
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

    clf = XGBClassifier(**param).fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(acc)
