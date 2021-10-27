import datetime
import os

from autofe.utils.logger import get_root_logger
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.tuner.optuna.tuner import OptunaTuner
from supervised.utils.metric import Metric

logger = get_root_logger(log_file=None)


def setup_outputdir_(path, warn_if_exist=True, create_dir=True):
    if path is None:
        for i in range(1, 1000):
            path = f'AutoTabular/optuna_{i}{os.path.sep}'
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError as e:
                print(e)
                path = f'AutoTabular/optuna_{i}{os.path.sep}'
        else:
            raise RuntimeError(
                'more than 1000 jobs launched in the same second')
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError as e:
            print(e)
            logger.warning(
                f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"'
            )
    path = os.path.expanduser(
        path)  # replace ~ with absolute path if it exists
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    return path


def setup_outputdir(path,
                    warn_if_exist=True,
                    create_dir=True,
                    path_suffix=None):
    if path_suffix is None:
        path_suffix = ''
    if path_suffix and path_suffix[-1] == os.path.sep:
        path_suffix = path_suffix[:-1]
    if path is not None:
        path = f'{path}{path_suffix}'
    if path is None:
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime('%Y%m%d_%H%M%S')
        path = f'AutogluonModels/optuna-{timestamp}{path_suffix}{os.path.sep}'
        for i in range(1, 1000):
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError as e:
                print(e)
                path = f'AutogluonModels/ag-{timestamp}-{i:03d}{path_suffix}{os.path.sep}'
        else:
            raise RuntimeError(
                'more than 1000 jobs launched in the same second')
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError as e:
            print(e)
            logger.warning(
                f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"'
            )
    path = os.path.expanduser(
        path)  # replace ~ with absolute path if it exists
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    return path


if __name__ == '__main__':
    setup_outputdir(path=None)
    x, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    eval_metric = Metric({'name': 'accuracy'})
    optuna_tuner = OptunaTuner(
        results_path='./',
        ml_task=MULTICLASS_CLASSIFICATION,
        eval_metric=eval_metric,
        max_trials=1,
    )
    best_params = optuna_tuner.optimize(
        algorithm='Random Forest',
        data_type='original',
        X_train=X_train,
        y_train=y_train,
        sample_weight=None,
        X_validation=X_test,
        y_validation=y_test,
        sample_weight_validation=None,
        learner_params={},
    )
    print(best_params)
