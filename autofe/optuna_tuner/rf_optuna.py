import numpy as np
import optuna
import pandas as pd
from autofe.optuna_tuner.registry import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION, default_optimizer_direction, default_task_metric, get_metric_fn, support_ml_task
from autofe.utils.logger import get_root_logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = get_root_logger(log_file=None)


class RandomForestOptuna(object):

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
            self.estimator = RandomForestRegressor
        else:
            self.estimator = RandomForestClassifier

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            split_ratio=0.2,
            max_evals: int = 100,
            timeout=3600):

        if X_val is not None:
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
            study = optuna.create_study(direction='maximize')
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

    def get_objective(self,
                      X_train,
                      y_train,
                      X_val=None,
                      y_val=None,
                      **kwargs):

        def objective(trial):
            if self.task == REGRESSION:
                criterion = trial.suggest_categorical(
                    'criterion',
                    ['squared_error', 'mse', 'absolute_error', 'poisson'])
            else:
                criterion = trial.suggest_categorical('criterion',
                                                      ['gini', 'entropy'])

            param = {
                'criterion':
                criterion,
                'min_samples_leaf':
                trial.suggest_int('min_samples_leaf', 1, 128),
                'min_samples_split':
                trial.suggest_int('min_samples_split', 2, 128),
                'max_depth':
                trial.suggest_int('max_depth', 2, 32),
                'max_features':
                trial.suggest_float('max_features', 0.01, 1),
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

        if len(set(tuning_data.columns)) < len(train_data.columns):
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
    X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    rf = RandomForestOptuna(task='multiclass_classification')
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    rf.fit(X_train, y_train, X_val=None, y_val=None, max_evals=10)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(acc)
