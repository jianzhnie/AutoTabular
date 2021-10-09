import sys

import numpy as np
from autogluon_benchmark.tasks import task_loader, task_utils

sys.path.append('../')

if __name__ == '__main__':
    task_dict = task_loader.get_task_dict()
    task_name = 'albert'  # task name in yaml config
    # task_id = task_dict[task_name]['openml_task_id']  # openml task id
    task_id = 189356  # albert
    n_folds = 1  # do 5 folds of train/val split

    fit_args = {
        'eval_metric': 'roc_auc',
        'hyperparameters': {
            'GBM': {
                'num_boost_round': 5
            },
            # 'NN': {'num_epochs': 5},
            'FASTAI': {},  # FIXME: This crashes during fillna
        },
        'time_limits': 300,
        'verbosity': 4,
    }

    predictors, scores = task_utils.run_task(
        task_id, n_folds=n_folds, fit_args=fit_args)
    score = float(np.mean(scores))
    if len(scores) > 1:
        score_std = np.std(scores, ddof=1)
    else:
        score_std = 0.0  # Should this be np.inf?
    print(f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})')
