# mljar-supervised 执行步骤

mljar-supervised 的AutoML的训练分为几个步骤。每个步骤表示在ML Pipeline 搜索性能最佳的机器学习模型的过程中常见的操作.

1. ==**simple_algorithms**==
2. ==**default_algorithms**==
3. ==**not_so_random**==
4. ==**golden_features**==
5. ==**features_selection**==
6. ==**hill_climbing**==
7. ==**ensemble**==
8. ==**stack**==
9. ==**ensemble_stacked**==


```python
def steps(self):

    all_steps = []
    if self._adjust_validation:
        all_steps += ["adjust_validation"]

    all_steps += ["simple_algorithms", "default_algorithms"]

    if self._start_random_models > 1:
        all_steps += ["not_so_random"]

    categorical_strategies = self._apply_categorical_strategies()
    if PreprocessingTuner.CATEGORICALS_MIX in categorical_strategies:
        all_steps += ["mix_encoding"]
    if PreprocessingTuner.CATEGORICALS_LOO in categorical_strategies:
        all_steps += ["loo_encoding"]
    if self._golden_features and self._can_apply_golden_features():
        all_steps += ["golden_features"]
    if self._kmeans_features and self._can_apply_kmeans_features():
        all_steps += ["kmeans_features"]
    if self._features_selection:
        all_steps += ["insert_random_feature"]
        all_steps += ["features_selection"]
    for i in range(self._hill_climbing_steps):
        all_steps += [f"hill_climbing_{i+1}"]
    if self._boost_on_errors:
        all_steps += ["boost_on_errors"]
    if self._train_ensemble:
        all_steps += ["ensemble"]
    if self._stack_models:
        all_steps += ["stack"]
        if self._train_ensemble:
            all_steps += ["ensemble_stacked"]
    return all_steps
```


# mljar-supervised 中的 [AutoML Modes](https://supervised.mljar.com/features/modes/#automl-modes)

There are 3 built-in modes available in AutoML:

- Explain - to be used when the user wants to explain and understand the data.
- Perform - to be used when the user wants to train a model that will be used in real-life use cases.
- Compete - To be used for machine learning competitions (maximum performance!).


## Custom modes
User can define his own modes by setting the parameters in AutoML constructor [AutoML API](https://supervised.mljar.com/features/modes/)

```python
AutoML(
    results_path=None,
    total_time_limit=60 * 60,
    mode="Explain",
    ml_task="auto",
    model_time_limit=None,
    algorithms="auto",
    train_ensemble=True,
    stack_models="auto",
    eval_metric="auto",
    validation_strategy="auto",
    explain_level="auto",
    golden_features="auto",
    features_selection="auto",
    start_random_models="auto",
    hill_climbing_steps="auto",
    top_models_to_improve="auto",
    boost_on_errors="auto",
    kmeans_features="auto",
    mix_encoding="auto",
    max_single_prediction_time=None,
    optuna_time_budget=None,
    optuna_init_params={},
    optuna_verbose=True,
    n_jobs=-1,
    verbose=1)
```

### 使用实例

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML

train = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/train.csv"
)
print(train.head())

X = train[train.columns[2:]]
y = train["Survived"]

automl = AutoML(
    algorithms=["CatBoost", "Xgboost", "LightGBM"],
    model_time_limit=30*60,
    start_random_models=10,
    hill_climbing_steps=3,
    top_models_to_improve=3,
    golden_features=True,
    features_selection=False,
    stack_models=True,
    train_ensemble=True,
    explain_level=0,
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 4,
        "shuffle": False,
        "stratify": True,
    }
)

automl.fit(X, y)
test = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/test_with_Survived.csv"
)
predictions = automl.predict(test)
print(predictions)
print(f"Accuracy: {accuracy_score(test['Survived'], predictions)*100.0:.2f}%")
```
- It will train models with CatBoost, Xgboost and LightGBM algorithms.
- Each model will be trained for 30 minutes (30*60 seconds). total_time_limit is not set.
- There will be trained about 10+ 3 * 3 * 2 = 28 unstacked models for each algorithm。
> 10 个 随机搜索的模型
> hill_climbing_steps *  top_models_to_improve *  golden_features or not = 3 * 3 * 2

- There will be trained about  10 stacked models for each algorithm. (There is stacked up to 10 models for each algorithm)

There will trained Ensemble based on unstacked models and Ensemble_Stacked from unstacked and stackd models.

In total there will be about 3*28+2=86 models trained.

explain_level=0 means that there will be only learning curves saved. No other explanations will be computed.


#  mljar-supervised 自定义算法

## 1. 自定义  data_preprocressing

以自定义一个 `labellencoder` 为例:
- 定义一个类 `LabelEncoder` 并进行初始化
- 定义 fit() 函数
- 定义 transform() 函数
- 定义 inverse_transform 函数


```python
class LabelEncoder(object):
    def __init__(self, try_to_fit_numeric=False):
        self.lbl = sk_preproc.LabelEncoder()
        self._try_to_fit_numeric = try_to_fit_numeric

    def fit(self, x):
        self.lbl.fit(x)  # list(x.values))
        if self._try_to_fit_numeric:
            logger.debug("Try to fit numeric in LabelEncoder")
            try:
                arr = {Decimal(c): c for c in self.lbl.classes_}
                sorted_arr = dict(sorted(arr.items()))
                self.lbl.classes_ = np.array(
                    list(sorted_arr.values()), dtype=self.lbl.classes_.dtype
                )
            except Exception as e:
                pass

    def transform(self, x):
        try:
            return self.lbl.transform(x)  # list(x.values))
        except ValueError as ve:
            # rescue
            classes = np.unique(x)  # list(x.values))
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(x)  # list(x.values))

    def inverse_transform(self, x):
        return self.lbl.inverse_transform(x)  # (list(x.values))

    def to_json(self):
        data_json = {}
        for i, cl in enumerate(self.lbl.classes_):
            data_json[str(cl)] = i
        return data_json

    def from_json(self, data_json):
        keys = np.array(list(data_json.keys()))
        if len(keys) == 2 and "False" in keys and "True" in keys:
            keys = [False, True]
        self.lbl.classes_ = keys
```

## 2. 自定义  data_preprocesing 的调用 

所有的 data_preprocesing 函数类都通过 [Preprocessing](../supervised/preprocessing/preprocessing.py) 进行调用

```python

class Preprocessing(object):
    def __init__(
        self,
        preprocessing_params={"target_preprocessing": [], "columns_preprocessing": {}},
        model_name=None,
        k_fold=None,
        repeat=None,
    ):

    def fit_and_transform(self, X_train, y_train, sample_weight=None):
        logger.debug("Preprocessing.fit_and_transform")

        if y_train is not None:
            # target preprocessing
            # this must be used first, maybe we will drop some rows because of missing target values
            target_preprocessing = self._params.get("target_preprocessing")
            logger.debug("target_preprocessing params: {}".format(target_preprocessing))

            X_train, y_train, sample_weight = ExcludeRowsMissingTarget.transform(
                X_train, y_train, sample_weight
            )

            if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
                logger.debug("Convert target to integer")
                self._categorical_y = LabelEncoder(try_to_fit_numeric=True)
                self._categorical_y.fit(y_train)
                y_train = pd.Series(self._categorical_y.transform(y_train))

            if PreprocessingCategorical.CONVERT_ONE_HOT in target_preprocessing:
                logger.debug("Convert target to one-hot coding")
                self._categorical_y = LabelBinarizer()
                self._categorical_y.fit(pd.DataFrame({"target": y_train}), "target")
                y_train = self._categorical_y.transform(
                    pd.DataFrame({"target": y_train}), "target"
                )

            if Scale.SCALE_LOG_AND_NORMAL in target_preprocessing:
                logger.debug("Scale log and normal")

                self._scale_y = Scale(
                    ["target"], scale_method=Scale.SCALE_LOG_AND_NORMAL
                )
                y_train = pd.DataFrame({"target": y_train})
                self._scale_y.fit(y_train)
                y_train = self._scale_y.transform(y_train)
                y_train = y_train["target"]

            if Scale.SCALE_NORMAL in target_preprocessing:
                logger.debug("Scale normal")

                self._scale_y = Scale(["target"], scale_method=Scale.SCALE_NORMAL)
                y_train = pd.DataFrame({"target": y_train})
                self._scale_y.fit(y_train)
                y_train = self._scale_y.transform(y_train)
                y_train = y_train["target"]
```


最后在 [ModelFramework](../supervised/model_framework.py) 中完成对数据预处理的调用


## Algorithms and AlgorithmsRegistry

在每个算法定义完成之后, 会通过 `AlgorithmsRegistry` 类对 Algorithms 信息进行注册. 例如

```python
def test_add_to_registry(self):
    class Model1:
        algorithm_short_name = ""

    model1 = {
        "task_name": "binary_classification",
        "model_class": Model1,
        "model_params": {},
        "required_preprocessing": {},
        "additional": {},
        "default_params": {},
    }

    AlgorithmsRegistry.add(**model1)
```

#### 调用 AlgorithmsRegistry 

```python
from supervised.algorithms.registry import AlgorithmsRegistry
model_info = AlgorithmsRegistry.registry["binary_classification"]["Xgboost"]
print(model_info)

{
'class': <class 'supervised.algorithms.xgboost.XgbAlgorithm'>, 

'params': {'objective': ['binary:logistic'], 'eta': [0.05, 0.075, 0.1, 0.15], 'max_depth': [4, 5, 6, 7, 8, 9], 'min_child_weight': [1, 5, 10, 25, 50], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, 'required_preprocessing': ['missing_values_inputation', 'convert_categorical', 'datetime_transform', 'text_transform', 'target_as_integer'], }, 

'additional': {'max_rounds': 10000, 'early_stopping_rounds': 50, 'max_rows_limit': None, 'max_cols_limit': None}, 

'default_params': {'objective': 'binary:logistic', 'eta': 0.075, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0}

}

```

#### 在 [MljarTuner](../supervised/tuner/mljar_tuner.py) 中调用 AlgorithmsRegistry

```python
## line 959
def _get_model_params(self, model_type, seed, params_type="random"):
    model_info = AlgorithmsRegistry.registry[self._ml_task][model_type]

    model_params = None
    if params_type == "default":

        model_params = model_info["default_params"]
        model_params["seed"] = seed

    else:
        model_params = RandomParameters.get(model_info["params"], seed + self._seed)
    if model_params is None:
        return None

    # set eval metric
    if model_info["class"].algorithm_short_name == "Xgboost":
        model_params["eval_metric"] = xgboost_eval_metric(
            self._ml_task, self._eval_metric
        )
    if model_info["class"].algorithm_short_name == "LightGBM":
        metric, custom_metric = lightgbm_eval_metric(
            self._ml_task, self._eval_metric
        )
        model_params["metric"] = metric
        model_params["custom_eval_metric_name"] = custom_metric
    if model_info["class"].algorithm_short_name == "CatBoost":
        model_params["eval_metric"] = catboost_eval_metric(
            self._ml_task, self._eval_metric
        )
    elif model_info["class"].algorithm_short_name in [
        "Random Forest",
        "Extra Trees",
    ]:
        model_params["eval_metric_name"] = self._eval_metric
        model_params["ml_task"] = self._ml_task

    required_preprocessing = model_info["required_preprocessing"]
    model_additional = model_info["additional"]
    preprocessing_params = PreprocessingTuner.get(
        required_preprocessing, self._data_info, self._ml_task
    )

    model_params = {
        "additional": model_additional,
        "preprocessing": preprocessing_params,
        "validation_strategy": self._validation_strategy,
        "learner": {
            "model_type": model_info["class"].algorithm_short_name,
            "ml_task": self._ml_task,
            "n_jobs": self._n_jobs,
            **model_params,
        },
        "automl_random_state": self._seed,
    }

    if self._data_info.get("num_class") is not None:
        model_params["learner"]["num_class"] = self._data_info.get("num_class")

    model_params["ml_task"] = self._ml_task
    model_params["explain_level"] = self._explain_level

    return model_params
```

通过  `model_info = AlgorithmsRegistry.registry[self._ml_task][model_type]` 可以获得每个模型的相关信息

