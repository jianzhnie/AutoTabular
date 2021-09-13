import pandas as pd
import numpy as np
from supervised.automl import AutoML
from sklearn.metrics import accuracy_score

train = pd.read_csv(
    "/Users/jianzhengnie/work/AutoTabular/autotabular/datasets/data/Titanic.csv")
print(train.head())

X = train[train.columns[2:]]
y = train["Survived"]

validation_strategy = {
    "validation_type": "split",
    "train_ratio": 0.8,
    "shuffle": True,
    "stratify": True
}

automl = AutoML(
    algorithms=["CatBoost", "Xgboost", "LightGBM"],
    model_time_limit=30 * 60,
    start_random_models=3,
    hill_climbing_steps=0,
    top_models_to_improve=3,
    golden_features=True,
    features_selection=True,
    stack_models=False,
    train_ensemble=True,
    explain_level=2,
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 2,
        "shuffle": True,
        "stratify": True,
    })

automl.fit(X, y)

test = pd.read_csv(
    "/Users/jianzhengnie/work/AutoTabular/autotabular/datasets/data/Titanic.csv")
predictions = automl.predict(test)
print(predictions)
print(f"Accuracy: {accuracy_score(test['Survived'], predictions)*100.0:.2f}%")
