import optuna
from sklearn.svm import SVC, SVR
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.utils.metric import Metric


class SVMObjective:
    def __init__(
        self,
        ml_task,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        eval_metric,
        random_state,
    ):
        self.ml_task = ml_task
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.seed = random_state

    def __call__(self, trial):
        try:
            Algorithm = (SVR if self.ml_task == REGRESSION else SVC)
            params = {
                'C':
                trial.suggest_float('C', 1e-10, 1e10, log=True),
                'kernel':
                trial.suggest_categorical(
                    'kernel',
                    ['poly', 'rbf']),
                # 'gamma':
                # trial.suggest_float('gamma', 1e-4, 1, log=True),
                'degree':
                trial.suggest_categorical('degree', [1, 2, 3, 4]),
                'class_weight': 'balanced',
                'random_state': self.seed,
                'probability': True,
                'shrinking': True,
            }
            model = Algorithm(**params).set_params(**params)
            model.fit(
                self.X_train, self.y_train, sample_weight=self.sample_weight)

            preds = model.predict(self.X_validation)

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print('Exception in SVMObjective', str(e))
            return None

        return score
