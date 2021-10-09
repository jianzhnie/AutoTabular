import numpy as np
from autogluon_benchmark.tasks import task_loader, task_utils


def run_task(task_name, fit_args=None):
    task_dict = task_loader.get_task_dict()
    task_id = task_dict[task_name]['openml_task_id']  # openml task id
    n_folds = 1  # do 5 folds of train/val split

    if fit_args is None:
        fit_args = dict()

    predictors, scores = task_utils.run_task(
        task_id, n_folds=n_folds, fit_args=fit_args, print_leaderboard=True)

    score = float(np.mean(scores))
    num_models = float(
        np.mean([len(predictor.get_model_names())
                 for predictor in predictors]))
    if len(scores) > 1:
        score_std = np.std(scores, ddof=1)
    else:
        score_std = 0.0  # Should this be np.inf?
    print(f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})')
    return score, num_models


if __name__ == '__main__':
    task_names = [
        'Australian',
        'blood-transfusion',
        'christine',
        'segment',
        'sylvine',
    ]

    fit_args = {
        'hyperparameters': {
            'GBM': {},
            'CAT': {},
            'NN': {}
        },
        'hyperparameter_tune': True,
        'time_limits': 60,
    }

    task_scores = dict()
    task_num_models = dict()
    for task_name in task_names:
        task_scores[task_name], task_num_models[task_name] = run_task(
            task_name=task_name, fit_args=fit_args)

    print('Scores:')
    print(task_scores)
    print('Num Models:')
    print(task_num_models)
