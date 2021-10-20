import json
import os
from copy import copy
from datetime import datetime
from pathlib import Path
from time import time
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Dataset as lgbDataset
from pytorch_widedeep.utils import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from lightgbm_optimizer import (  # isort:skipimport pickle  # noqa: E402
    LGBOptimizerHyperopt, LGBOptimizerOptuna,
)

SEED = 42
pd.options.display.max_columns = 100

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/fb_comments/'
    RESULTS_DIR = ROOTDIR / 'results/fb_comments/lightgbm'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    fb_comments = pd.read_csv(PROCESSED_DATA_DIR / 'fb_comments.csv')
    target_name = 'target'

    OPTIMIZE_WITH = 'optuna'

    cat_cols = []
    for col in fb_comments.columns:
        if fb_comments[col].dtype == 'O' or fb_comments[col].nunique(
        ) < 200 and col != 'target':
            cat_cols.append(col)
    num_cols = [
        c for c in fb_comments.columns if c not in cat_cols + ['target']
    ]

    #  TRAIN/VALID for hyperparam optimization
    label_encoder = LabelEncoder(cat_cols)
    fb_comments = label_encoder.fit_transform(fb_comments)

    X = fb_comments.drop(target_name, axis=1)
    y = fb_comments[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    lgbtrain = lgbDataset(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    lgbvalid = lgbDataset(
        X_test,
        y_test,
        reference=lgbtrain,
        free_raw_data=False,
    )

    if OPTIMIZE_WITH == 'optuna':
        optimizer: Union[LGBOptimizerHyperopt,
                         LGBOptimizerOptuna] = LGBOptimizerOptuna(
                             objective='regression')
    elif OPTIMIZE_WITH == 'hyperopt':
        optimizer = LGBOptimizerHyperopt(objective='regression', verbose=True)

    optimizer.optimize(lgbtrain, lgbvalid)

    # Final TRAIN/TEST
    params = copy(optimizer.best)
    params['n_estimators'] = 1000

    flgbtrain = lgbDataset(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    lgbtest = lgbDataset(
        X_test,
        y_test,
        reference=flgbtrain,
        free_raw_data=False,
    )

    start = time()
    model = lgb.train(
        params,
        flgbtrain,
        valid_sets=[lgbtest],
        early_stopping_rounds=50,
        verbose_eval=True,
    )
    runtime = time() - start

    preds = model.predict(lgbtest.data)
    rmse = np.sqrt(mean_squared_error(lgbtest.label, preds))
    r2 = r2_score(lgbtest.label, preds)
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')

    # SAVE
    suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
    results_filename = '_'.join(['fb_comments_lightgbm', suffix]) + '.json'
    results_d = {}
    results_d['best_params'] = optimizer.best
    results_d['runtime'] = runtime
    results_d['rmse'] = rmse
    results_d['r2'] = r2
    with open(RESULTS_DIR / results_filename, 'w') as f:
        json.dump(results_d, f, indent=4)
