import json
import os
import pickle
from copy import copy
from datetime import datetime
from pathlib import Path
from time import time
from typing import Union

import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset as lgbDataset
from lightgbm_optimizer import LGBOptimizerHyperopt, LGBOptimizerOptuna
from pytorch_widedeep.utils import LabelEncoder
from sklearn.metrics import accuracy_score

pd.options.display.max_columns = 100

ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

RESULTS_DIR = ROOTDIR / 'results/adult/lightgbm'
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

MODELS_DIR = ROOTDIR / 'results/adult/models/lightgbm'
if not MODELS_DIR.is_dir():
    os.makedirs(MODELS_DIR)

OPTIMIZE_WITH = 'hyperopt'

train = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_train.pkl')
valid = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_val.pkl')
test = pd.read_pickle(PROCESSED_DATA_DIR / 'adult_test.pkl')
for df in [train, valid, test]:
    df.drop('education_num', axis=1, inplace=True)

cat_cols = []
for col in train.columns:
    if train[col].dtype == 'O' or train[col].nunique(
    ) < 200 and col != 'target':
        cat_cols.append(col)

#  TRAIN/VALID for hyperparam optimization
label_encoder = LabelEncoder(cat_cols)
train_le = label_encoder.fit_transform(train)
valid_le = label_encoder.transform(valid)

lgbtrain = lgbDataset(
    train_le[cat_cols],
    train_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbvalid = lgbDataset(
    valid_le[cat_cols],
    valid_le.target,
    reference=lgbtrain,
    free_raw_data=False)

if OPTIMIZE_WITH == 'optuna':
    optimizer: Union[LGBOptimizerHyperopt,
                     LGBOptimizerOptuna] = LGBOptimizerOptuna()
elif OPTIMIZE_WITH == 'hyperopt':
    optimizer = LGBOptimizerHyperopt(verbose=True)

optimizer.optimize(lgbtrain, lgbvalid)

# Final TRAIN/TEST
ftrain = pd.concat([train, valid]).reset_index(drop=True)
flabel_encoder = LabelEncoder(cat_cols)
ftrain_le = flabel_encoder.fit_transform(ftrain)
test_le = flabel_encoder.transform(test)

params = copy(optimizer.best)
params['n_estimators'] = 1000

flgbtrain = lgbDataset(
    ftrain_le[cat_cols],
    ftrain_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbtest = lgbDataset(
    test_le[cat_cols],
    test_le.target,
    reference=flgbtrain,
    free_raw_data=False)

start = time()
model = lgb.train(
    params,
    flgbtrain,
    valid_sets=[lgbtest],
    early_stopping_rounds=50,
    verbose_eval=True,
)
runtime = time() - start

preds = (model.predict(lgbtest.data) > 0.5).astype('int')
acc = accuracy_score(lgbtest.label, preds)
print(f'Accuracy: {acc}')

# SAVE
suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
results_filename = '_'.join(['adult_lightgbm', suffix]) + '.json'
results_d = {}
results_d['best_params'] = optimizer.best
results_d['runtime'] = runtime
results_d['acc'] = acc
with open(RESULTS_DIR / results_filename, 'w') as f:
    json.dump(results_d, f, indent=4)

model_filename = '_'.join(['model_adult_lightgbm', suffix]) + '.pkl'
with open(MODELS_DIR / model_filename, 'wb') as f:
    pickle.dump(model, f)
