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
        sample_weight_validation,
        eval_metric,
        n_jobs,
        random_state,
    ):
        self.ml_task = ml_task
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.objective = 'mse' if ml_task == REGRESSION else 'gini'
        self.max_steps = 10  # RF is trained in steps 100 trees each
        self.seed = random_state

    def __call__(self, trial):
        try:
            Algorithm = (SVR if self.ml_task == REGRESSION else SVC)
            params = {
                'C':
                trial.suggest_float('C', 1e-3, 1024, log=True),
                'kernel':
                trial.suggest_categorical(
                    'kernel',
                    ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
                'gamma':
                trial.suggest_float('gamma', 1e-4, 1e-1),
                'degree':
                trial.suggest_categorical('degree', [0, 1, 2, 3]),
                'class_weight':
                trial.suggest_categorical('class_weight', ['balanced', None]),
                'n_jobs':
                self.n_jobs,
                'random_state':
                self.seed
            }
            model = Algorithm(params)
            model.fit(
                self.X_train, self.y_train, sample_weight=self.sample_weight)

            preds = model.predict(self.X_validation)

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print('Exception in RandomForestObjective', str(e))
            return None

        return score
