import sys
import time

import numpy as np
import openml
from autogluon_benchmark.tasks import task_loader, task_utils
from openml.exceptions import OpenMLServerException

sys.path.append('../')


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


if __name__ == '__main__':
    task_dict = task_loader.get_task_dict()
    task_name = 'adult'  # task name in yaml config
    task_id = task_dict[task_name]['openml_task_id']  # openml task id
    n_folds = 5  # do 5 folds of train/val split

    fit_args = {
        'eval_metric': 'roc_auc',
    }
    task = task_id
    if isinstance(task, int):
        task_id = task
        delay_exp = 0
        while True:
            try:
                print(f'Getting task {task_id}')
                task = openml.tasks.get_task(task_id)
                print(f'Got task {task_id}')
            except OpenMLServerException as e:
                delay = 2**delay_exp
                delay_exp += 1
                if delay_exp > 10:
                    raise ValueError('Unable to get task after 10 retries')
                print(e)
                print(f'Retry in {delay}s...')
                time.sleep(delay)
                continue
            break

    n_repeats_full, n_folds_full, n_samples_full = task.get_split_dimensions()

    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    print(type(X))
    print(type(y))

    predictors, scores = task_utils.run_task(
        task_id, n_folds=n_folds, fit_args=fit_args)
    score = float(np.mean(scores))
    if len(scores) > 1:
        score_std = np.std(scores, ddof=1)
    else:
        score_std = 0.0  # Should this be np.inf?
    print(f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})')
