from autogluon_benchmark.tasks.task_loader import get_task_dict
from autogluon_benchmark.tasks.task_utils import get_task
from autogluon_benchmark.utils import data_utils

if __name__ == '__main__':
    task_dict = get_task_dict()

    task_names = list(task_dict.keys())

    task_metadatas = {}
    for task_name in task_names:
        task = get_task(task_dict[task_name]['openml_task_id'])
        X, y, _, _ = task.get_dataset().get_data(task.target_name)
        task_metadata = data_utils.get_data_metadata(X, y)
        task_metadatas[task_name] = task_metadata

    import pprint
    pprint.pprint(task_metadatas)
