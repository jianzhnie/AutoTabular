import numpy as np
import pandas as pd
from autotabular.algorithms.deepnets import *
from autotabular.datasets import dsutils
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = dsutils.load_bank()
    df.drop(['id'], axis=1, inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y = df_train.pop('y')
    y_test = df_test.pop('y')


    config = deeptable.ModelConfig(nets=deepnets.DeepFM, auto_discrete=True, metrics=['AUC'])
    dt = deeptable.DeepTable(config=config)

    model, history = dt.fit(df_train, y, epochs=10)

    proba = dt.predict_proba(df_test)
    preds = dt.predict(df_test)

    result = dt.evaluate(df_test, y_test, batch_size=512, verbose=0)