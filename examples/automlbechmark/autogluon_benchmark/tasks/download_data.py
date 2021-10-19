import sys
import pandas as pd
import yaml

sys.path.append('../../')
from autogluon_benchmark.tasks.task_utils import *

if __name__ == "__main__":
    yaml_path = "/home/wenqi-ao/userdata/workdirs/autotabular/examples/automlbechmark/autogluon_benchmark/tasks/small.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = f.read()
        d = yaml.load(cfg)
    for item in d:
        task_id = item['openml_task_id']
        # if task_id == 3917:
        task = get_task(task_id)
        X, y = get_dataset(task)
        data = pd.concat([X, y], axis = 1)
        print(f"task name: {item['name']}")
        print(data.info())
            # break