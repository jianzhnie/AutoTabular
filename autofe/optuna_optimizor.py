from typing import Dict

import optuna
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


class RandomForestHpo(object):

    def __init__(self,
                 task: str = 'binary',
                 metric: str = 'accuracy',
                 verbose: bool = False):

        self.task = task
        self.metric = metric
        self.verbose = verbose
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
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_evals, timeout=timeout)
        trial = study.best_trial
        best_param = trial.params
        print()
        print('Finished  HPO training, get the best model params: ')
        print(best_param)
        print('Retraining on the whole dataset')
        self.model = self.estimator(**best_param).fit(X_train, y_train)
        return best_param

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return preds

    def predict_proba(self, X_test):
        preds = self.model.predict_proba(X_test)
        return preds

    def get_score_fn(self, task):
        score_dict = {
            'binary': roc_auc_score,
            'multiclass': accuracy_score,
            'regression': r2_score,
        }
        return score_dict[task]

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
            if self.task == 'regression':
                estimator = RandomForestRegressor(**param)
            else:
                estimator = RandomForestClassifier(**param)

            model = estimator.fit(X_train, y_train)
            preds = model.predict(X_val)
            score_fn = self.get_score_fn(self.task)
            score = score_fn(y_val, preds)

            return score

        return objective


if __name__ == '__main__':
    x, y = sklearn.datasets.load_iris(return_X_y=True)
    rf = RandomForestClassifier().fit(x, y)
    print(rf.predict(x))
    rf = RandomForestHpo(task='multiclass')
    rf.fit(x, y)
    print(rf.predict(x))
