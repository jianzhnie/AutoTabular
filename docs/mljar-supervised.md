
#  mljar-supervised 自定义算法

## 1. 自定义  data_preprocressing

以自定义一个 `labellencoder` 为例:
- 定义一个 类 `LabelEncoder` 并进行初始化
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

### AutoML api 调用, 参考 [AutoML](https://supervised.mljar.com/features/modes/)

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

# automl = AutoML(mode="Compete") # default mode is Explain
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
        "k_folds": 2,
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
