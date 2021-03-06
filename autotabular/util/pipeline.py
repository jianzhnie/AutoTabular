from typing import Any, Dict, List, Optional, Tuple

from autotabular.constants import (BINARY_CLASSIFICATION, CLASSIFICATION_TASKS,
                                   MULTICLASS_CLASSIFICATION,
                                   MULTILABEL_CLASSIFICATION,
                                   MULTIOUTPUT_REGRESSION, REGRESSION,
                                   REGRESSION_TASKS)
from autotabular.pipeline.classification import SimpleClassificationPipeline
from autotabular.pipeline.regression import SimpleRegressionPipeline
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.pipeline import Pipeline

__all__ = [
    'get_configuration_space',
    'get_class',
]


def parse_include_exclude_components(
    task: int,
    include_estimators: Optional[List[str]] = None,
    exclude_estimators: Optional[List[str]] = None,
    include_preprocessors: Optional[List[str]] = None,
    exclude_preprocessors: Optional[List[str]] = None
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    exclude = dict()
    include = dict()
    if include_preprocessors is not None and \
            exclude_preprocessors is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

    if include_estimators is not None and \
            exclude_estimators is not None:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif include_estimators is not None:
        if task in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif task in REGRESSION_TASKS:
            include['regressor'] = include_estimators
        else:
            raise ValueError(task)
    elif exclude_estimators is not None:
        if task in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif task in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators
        else:
            raise ValueError(task)
    return include, exclude


def get_configuration_space(
        info: Dict[str, Any],
        include_estimators: Optional[List[str]] = None,
        exclude_estimators: Optional[List[str]] = None,
        include_preprocessors: Optional[List[str]] = None,
        exclude_preprocessors: Optional[List[str]] = None
) -> ConfigurationSpace:
    include, exclude = parse_include_exclude_components(
        task=info['task'],
        include_estimators=include_estimators,
        exclude_estimators=exclude_estimators,
        include_preprocessors=include_preprocessors,
        exclude_preprocessors=exclude_preprocessors,
    )

    if info['task'] in REGRESSION_TASKS:
        return _get_regression_configuration_space(info, include, exclude)
    else:
        return _get_classification_configuration_space(info, include, exclude)


def _get_regression_configuration_space(
        info: Dict[str, Any], include: Dict[str, List[str]],
        exclude: Dict[str, List[str]]) -> ConfigurationSpace:
    task_type = info['task']
    sparse = False
    multioutput = False
    if task_type == MULTIOUTPUT_REGRESSION:
        multioutput = True

    dataset_properties = {'multioutput': multioutput, 'sparse': sparse}

    if info['is_sparse'] == 1:
        sparse = True
    configuration_space = SimpleRegressionPipeline(
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude).get_hyperparameter_search_space()
    return configuration_space


def _get_classification_configuration_space(
        info: Dict[str, Any], include: Dict[str, List[str]],
        exclude: Dict[str, List[str]]) -> ConfigurationSpace:
    task_type = info['task']

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
    if task_type == REGRESSION:
        raise NotImplementedError()
    if task_type == MULTICLASS_CLASSIFICATION:
        multiclass = True
    if task_type == BINARY_CLASSIFICATION:
        pass

    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {
        'multilabel': multilabel,
        'multiclass': multiclass,
        'sparse': sparse
    }

    return SimpleClassificationPipeline(
        dataset_properties=dataset_properties,
        include=include, exclude=exclude).\
        get_hyperparameter_search_space()


def get_class(info: Dict[str, Any]) -> Pipeline:
    if info['task'] in REGRESSION_TASKS:
        return SimpleRegressionPipeline
    else:
        return SimpleClassificationPipeline
