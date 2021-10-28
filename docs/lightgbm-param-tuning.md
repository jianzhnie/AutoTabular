

## [LightGBM](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/D1_TVqAXBauazCJdXJ8cpQ)参数概述

通常，基于树的模型的超参数可以分为 4 类：

1. 影响决策树结构和学习的参数
2. 影响训练速度的参数
3. 提高精度的参数
4. 防止过拟合的参数

大多数时候，这些类别有很多重叠，提高一个类别的效率可能会降低另一个类别的效率。如果完全靠手动调参，那会比较痛苦。**所以前期我们可以利用一些自动化调参工具给出一个大致的结果，而自动调参工具的核心在于如何给定适合的参数区间范围。** 如果能给定合适的参数网格，`Optuna` 就可以自动找到这些类别之间最平衡的参数组合。

下面对`LGBM`的4类超参进行介绍。

## 1、控制树结构的超参数

**max_depth 和 num_leaves**

在 `LGBM` 中，控制树结构的最先要调的参数是`max_depth`（树深度） 和 `num_leaves`（叶子节点数）。这两个参数对于树结构的控制最直接了断，因为 `LGBM` 是 `leaf-wise` 的，如果不控制树深度，会非常容易过拟合。`max_depth`一般设置可以尝试设置为`3到8`。

这两个参数也存在一定的关系。由于是二叉树，`num_leaves`最大值应该是`2^(max_depth)`。所以，确定了`max_depth`也就意味着确定了`num_leaves`的取值范围。

**min_data_in_leaf**

树的另一个重要结构参数是`min_data_in_leaf`，它的大小也与是否过拟合有关。它指定了叶子节点向下分裂的的最小样本数，比如设置100，那么如果节点样本数量不够100就停止生长。当然，`min_data_in_leaf`的设定也取决于训练样本的数量和`num_leaves`。对于大数据集，一般会设置千级以上。

## 2、提高准确性的超参数

**learning_rate 和 n_estimators**

实现更高准确率的常见方法是使用更多棵子树并降低学习率。换句话说，就是要找到`LGBM`中`n_estimators`和`learning_rate`的最佳组合。

`n_estimators`控制决策树的数量，而`learning_rate`是梯度下降的步长参数。经验来说，`LGBM` 比较容易过拟合，`learning_rate`可以用来控制梯度提升学习的速度，一般值可设在 `0.01 和 0.3` 之间。一般做法是先用稍多一些的子树比如1000，并设一个较低的`learning_rate`，然后通过`early_stopping`找到最优迭代次数。

**max_bin**

除此外，也可以增加`max_bin`(默认值为255)来提高准确率。因为变量分箱的数量越多，信息保留越详细，相反，变量分箱数量越低，信息越损失，但更容易泛化。这个和特征工程的分箱是一个道理，只不过是通过内部的`hist`直方图算法处理了。如果`max_bin`过高，同样也存在过度拟合的风险。

## 3、更多超参数来控制过拟合

**lambda_l1 和 lambda_l2**

`lambda_l1` 和 `lambda_l2` 对应着 `L1` 和 `L2` 正则化，和 `XGBoost` 的 `reg_lambda` 和 `reg_alpha` 是一样的，对叶子节点数和叶子节点权重的惩罚，值越高惩罚越大。这些参数的最佳值更难调整，因为它们的大小与过拟合没有直接关系，但会有影响。一般的搜索范围可以在 `(0, 100)`。

**min_gain_to_split**

这个参数定义着分裂的最小增益。这个参数也看出数据的质量如何，计算的增益不高，就无法向下分裂。如果你设置的深度很深，但又无法向下分裂，`LGBM`就会提示`warning`，无法找到可以分裂的了，说明数据质量已经达到了极限了。参数含义和 `XGBoost` 的 `gamma` 是一样。比较保守的搜索范围是 `(0, 20)`，它可以用作大型参数网格中的额外正则化。

**bagging_fraction 和 feature_fraction**

这两个参数取值范围都在`(0,1)`之间。

`feature_fraction`指定训练每棵树时要采样的特征百分比，它存在的意义也是为了避免过拟合。因为有些特征增益很高，可能造成每棵子树分裂的时候都用同一个特征，这样每个子树就同质化了。而如果通过较低概率的特征采样，可以避免每次都遇到这些强特征，从而让子树的特征变得差异化，即泛化。

`bagging_fraction`指定用于训练每棵树的训练样本百分比。要使用这个参数，还需要设置 `bagging_freq`，道理和`feature_fraction`一样，也是让没棵子树都变得**好而不同**。

## 4、在 Optuna 中创建搜索网格

`Optuna` 中的优化过程首先需要一个目标函数，该函数里面包括：

- 字典形式的参数网格
- 创建一个模型（可以配合交叉验证`kfold`）来尝试超参数组合集
- 用于模型训练的数据集
- 使用此模型生成预测
- 根据用户定义的指标对预测进行评分并返回

下面给出一个常用的框架，模型是5折的`Kfold`，这样可以保证模型的稳定性。最后一行返回了需要优化的 CV 分数的平均值。目标函数可以自己设定，比如指标`logloss`最小，`auc`最大，`ks`最大，训练集和测试集的`auc`差距最小等等。

```python
import optuna  # pip install optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

def objective(trial, X, y):
    # 后面填充
    param_grid = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = preds

    return np.mean(cv_scores)
```

下面是参数的设置，`Optuna`比较常见的方式`suggest_categorical`，`suggest_int`，`suggest_float`。其中，`suggest_int`和`suggest_float`的设置方式为`(参数，最小值，最大值，step=步长)`。

```python
def objective(trial, X, y):
    # 字典形式的参数网格
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }
```

## 5、创建 Optuna 自动调起来

下面是完整的目标函数框架，供参考：

```python
from optuna.integration import LightGBMPruningCallback

def objective(trial, X, y):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        "random_state": 2021,
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LGBM建模
        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],
        )
        # 模型预测
        preds = model.predict_proba(X_test)
        # 优化指标logloss最小
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)
```

上面这个网格里，还添加了`LightGBMPruningCallback`，这个`callback`类很方便，它可以在对数据进行训练之前检测出不太好的超参数集，从而显着减少搜索时间。

设置完目标函数，现在让参数调起来！

```python
study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=20)
```

`direction`可以是`minimize`，也可以是`maximize`，比如让`auc`最大化。然后可以设置`trials`来控制尝试的次数，理论上次数越多结果越优，但也要考虑下运行时间。

搜索完成后，调用`best_value`和`bast_params`属性，调参就出来了。