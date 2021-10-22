from typing import Dict

import numpy as np
import optuna
import pandas as pd
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from autofe.utils.logger import get_root_logger
logger = get_root_logger(log_file=None)


class RandomForestHpo(object):

    def __init__(self,
                 target: str = None,
                 task: str = 'binary',
                 metric: str = 'accuracy'):

        self.target = target
        self.task = task
        self.metric = metric
        self.early_stop_dict: Dict = {}

        if self.task == 'regression':
            self.estimator = RandomForestRegressor
        else:
            self.estimator = RandomForestClassifier

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            max_evals: int = 10,
            timeout=600):
        objective = self.get_objective(X_train, y_train, X_val, y_val)
        logger.info('===== Beginning  RandomForest Hpo training ======')
        logger.info('Max Hpo trials:  %s ' % max_evals)
        logger.info('Time Out: %s s ' % timeout)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_evals, timeout=timeout)
        trial = study.best_trial
        best_param = trial.params
        logger.info('====== Finished RandomForest Hpo training ======')
        logger.info('Get the best model params ...')
        logger.info('parms: %s', best_param)
        logger.info('Retraining on the whole dataset.')
        self.model = self.estimator(**best_param).fit(X_train, y_train)
        return best_param

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return preds

    def predict_proba(self, X_test):
        preds = self.model.predict_proba(X_test)
        return preds

    def get_score_fn(self, task, metric):
        support_metric_dict = {
            'auc': roc_auc_score,
            'accuracy': accuracy_score,
            'r2': r2_score,
            'mse': mean_squared_error
        }
        default_score_dict = {
            'binary': roc_auc_score,
            'multiclass': accuracy_score,
            'regression': r2_score,
        }
        if metric is not None:
            score_fn = support_metric_dict[metric]
        else:
            score_fn = default_score_dict[task]
        return score_fn

    def get_objective(self,
                      X_train,
                      y_train,
                      X_val=None,
                      y_val=None,
                      **kwargs):
        assert X_train is not None, 'train data most exist'
        assert y_train is not None, 'label data most exist'
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.20, random_state=42)

        def objective(trial):
            param = {
                'criterion':
                trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'min_samples_leaf':
                trial.suggest_int('min_samples_leaf', 1, 20),
                'min_samples_split':
                trial.suggest_int('min_samples_split', 2, 20),
                'max_depth':
                trial.suggest_int('max_depth', 2, 12),
                'n_estimators':
                trial.suggest_int('n_estimators', 100, 1000, step=100),
            }

            model = self.estimator(**param).fit(X_train, y_train)
            preds = model.predict(X_val)
            score_fn = self.get_score_fn(self.task, self.metric)
            score = score_fn(y_val, preds)

            return score

        return objective

    def _validate_fit_data(self, train_data, tuning_data=None):
        if not isinstance(train_data, pd.DataFrame):
            raise AssertionError(
                f'train_data is required to be a pandas DataFrame, but was instead: {type(train_data)}'
            )

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )
        if tuning_data is not None:
            if not isinstance(tuning_data, pd.DataFrame):
                raise AssertionError(
                    f'tuning_data is required to be a pandas DataFrame, but was instead: {type(tuning_data)}'
                )
            train_features = [
                column for column in train_data.columns
                if column != self.target
            ]
            tuning_features = [
                column for column in tuning_data.columns
                if column != self.target
            ]
            train_features = np.array(train_features)
            tuning_features = np.array(tuning_features)
            if np.any(train_features != tuning_features):
                raise ValueError(
                    'Column names must match between training and tuning data')
        return train_data, tuning_data


if __name__ == '__main__':
    x, y = sklearn.datasets.load_iris(return_X_y=True)
    rf = RandomForestHpo(target=None, task='multiclass')
    rf.fit(x, y)
    preds = rf.predict(x)
    acc = accuracy_score(y, preds)
    print(acc)
