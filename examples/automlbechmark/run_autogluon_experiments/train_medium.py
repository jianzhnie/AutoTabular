import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from autogluon_benchmark.tasks import task_transformer_utils, task_utils

sys.path.append('../')


def get_task_dict(yaml_files=None):
    if yaml_files is None:
        yaml_files = ['medium.yaml']
    parent_dir = Path(
        '/home/robin/jianzh/autotabular/examples/automlbechmark/autogluon_benchmark/tasks'
    )
    task_dict_full = {}
    for yaml_file in yaml_files:
        yaml_path = Path.joinpath(parent_dir, yaml_file)
        with open(yaml_path, 'r') as stream:
            task_list = yaml.load(stream, Loader=yaml.Loader)
        task_dict = {d['name']: d for d in task_list}
        for task in task_dict.values():
            task.pop('name')
        for key in task_dict:
            if key in task_dict_full:
                raise KeyError('Multiple yaml files contain the same key!')
        task_dict_full.update(task_dict)
    return task_dict_full


if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    RESULTS_DIR = ROOTDIR / 'results/medium_data/'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    task_dict = get_task_dict()
    for task_name in task_dict:  # task name in yaml config
        task_id = task_dict[task_name]['openml_task_id']  # openml task id

        n_folds = 2  # do 5 folds of train/val split
        init_args = {
            'eval_metric': 'acc',
        }

        fit_args = {
            'time_limit': 7200,
            'num_bag_folds': 5,
            'num_stack_levels': 1,
            'num_bag_sets': 1,
            'verbosity': 2,
        }

        try:
            predictors, scores, eval_dict = task_utils.run_task(
                task_id,
                n_folds=n_folds,
                init_args=init_args,
                fit_args=fit_args)
            score = float(np.mean(scores))
            if len(scores) > 1:
                score_std = np.std(scores, ddof=1)
            else:
                score_std = 0.0  # Should this be np.inf?
            print(
                f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})'
            )

            suffix = 'autogluon'
            results_filename = '_'.join([
                task_name,
                suffix,
            ]) + '.json'
            with open(RESULTS_DIR / results_filename, 'w') as f:
                json.dump(eval_dict, f, indent=4)
        except:
            pass

        try:
            predictors, scores, eval_dict = task_transformer_utils.run_task(
                task_id,
                n_folds=n_folds,
                init_args=init_args,
                fit_args=fit_args)
            score = float(np.mean(scores))
            if len(scores) > 1:
                score_std = np.std(scores, ddof=1)
            else:
                score_std = 0.0  # Should this be np.inf?
            print(
                f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})'
            )
            suffix = 'autogluon_transformer'
            results_filename = '_'.join([task_name, suffix]) + '.json'
            with open(RESULTS_DIR / results_filename, 'w') as f:
                json.dump(eval_dict, f, indent=4)
        except:
            pass
