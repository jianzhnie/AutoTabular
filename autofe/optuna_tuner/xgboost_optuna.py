import numpy as np
import optuna
import pandas as pd
from autofe.optuna_tuner.registry import (BINARY_CLASSIFICATION,
                                          MULTICLASS_CLASSIFICATION,
                                          REGRESSION,
                                          default_optimizer_direction,
                                          default_task_metric, get_metric_fn,
                                          support_ml_task)
from autofe.utils.logger import get_root_logger
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier, XGBRegressor

logger = get_root_logger(log_file=None)


class XGBoostOptuna(object):

    def __init__(
        self,
        task: str = BINARY_CLASSIFICATION,
        metric: str = 'accuracy',
        random_state=None,
    ):

        self.task = task
        self.metric = metric
        self.seed = random_state

        assert self.task in support_ml_task, 'Only Support ML Tasks: %s' % support_ml_task

        if self.task == REGRESSION:
            self.estimator = XGBRegressor
        else:
            self.estimator = XGBClassifier

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            split_ratio=0.2,
            max_evals: int = 100,
            timeout=600):

        X_train, X_val = self._validate_fit_data(
            train_data=X_train, tuning_data=X_val)

        if X_val is None:
            logger.info(
                'Tuning data is None, the original train_data will be split: train vs val =  %2s vs %2s'
                % (1 - split_ratio, split_ratio))
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=split_ratio)

        objective = self.get_objective(X_train, y_train, X_val, y_val)
        logger.info('===== Beginning  RandomForest Hpo training ======')
        logger.info('Max Hpo trials:  %s' % max_evals)
        logger.info('Time Out: %s s ' % timeout)

        try:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study = optuna.create_study(pruner=pruner, direction='maximize')
            study.optimize(objective, n_trials=max_evals, timeout=timeout)
            trial = study.best_trial
            best_param = trial.params
            logger.info('====== Finished RandomForest Hpo training ======')
            logger.info('Get the best model params ...')
            logger.info('parms: %s', best_param)
            logger.info('Retraining on the whole dataset.')
            self.model = self.estimator(**best_param).fit(X_train, y_train)

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print('Exception in RandomForestObjective', str(e))
            return None
        return best_param

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_score_fn(self, task, metric):
        if metric is None:
            metric = default_task_metric[task]
        score_fn = get_metric_fn[metric]
        return score_fn

    def get_optimizer_direction(self, task, metric):
        if metric is not None:
            metric = default_task_metric[task]
        direction = default_optimizer_direction[task]
        return direction

    def xgboost_objective(self, ml_task):
        objective = 'reg:squarederror'
        if ml_task == BINARY_CLASSIFICATION:
            objective = 'binary:logistic'
        elif ml_task == MULTICLASS_CLASSIFICATION:
            objective = 'multi:softprob'
        else:  # ml_task == REGRESSION
            objective = 'reg:squarederror'
        return objective

    def get_objective(self,
                      X_train,
                      y_train,
                      X_val=None,
                      y_val=None,
                      **kwargs):

        def objective(trial):
            obj = self.xgboost_objective(self.task)
            param = {
                'verbosity':
                0,
                'objective':
                obj,
                'use_label_encoder':
                False,
                'booster':
                trial.suggest_categorical('booster',
                                          ['gbtree', 'gblinear', 'dart']),
                'learning_rate':
                trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth':
                trial.suggest_int('max_depth', 2, 32, step=1),
                'n_estimators':
                trial.suggest_int('n_estimators', 100, 1000, step=100),
                'subsample':
                trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree':
                trial.suggest_float('colsample_bytree', 0.2, 1.0),
                'lambda':
                trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha':
                trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            }

            if param['booster'] == 'gbtree' or param['booster'] == 'dart':
                param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
                param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
                param['min_child_weight'] = trial.suggest_int(
                    'min_child_weight', 2, 10)
                param['gamma'] = trial.suggest_float(
                    'gamma', 1e-8, 1.0, log=True)
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
            train_features = train_data.columns
            tuning_features = tuning_data.columns
            train_features = np.array(train_features)
            tuning_features = np.array(tuning_features)
            if np.any(train_features != tuning_features):
                raise ValueError(
                    'Column names must match between training and tuning data')
        return train_data, tuning_data


if __name__ == '__main__':
    import sklearn.datasets
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    rf = XGBoostOptuna(task='multiclass_classification')
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    rf.fit(X_train, y_train, X_val=None, y_val=None, max_evals=10)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(acc)
