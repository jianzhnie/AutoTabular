from supervised.algorithms.registry import AlgorithmsRegistry


model_info = AlgorithmsRegistry.registry["binary_classification"]["Xgboost"]


if model_info["class"].algorithm_short_name == "Xgboost":
    print(model_info["class"].algorithm_short_name)
