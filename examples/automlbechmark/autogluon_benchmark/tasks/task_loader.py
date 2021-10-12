from pathlib import Path

import yaml


def get_task_dict(yaml_files=None):
    if yaml_files is None:
        yaml_files = ['small.yaml']
    parent_dir = Path(__file__).resolve().parent
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
