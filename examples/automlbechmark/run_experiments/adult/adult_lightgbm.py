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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
pd.options.display.max_columns = 100

ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

RESULTS_DIR = ROOTDIR / 'results/adult/lightgbm'
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

MODELS_DIR = ROOTDIR / 'results/adult/models/lightgbm'
if not MODELS_DIR.is_dir():
    os.makedirs(MODELS_DIR)

OPTIMIZE_WITH = 'optuna'

adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
target_name = 'target'

cat_col_names = []
for col in adult_data.columns:
    if adult_data[col].dtype == 'object':
        cat_col_names.append(col)

label_encoder = LabelEncoder(cat_col_names)
adult_data = label_encoder.fit_transform(adult_data)

X = adult_data.drop(target_name, axis=1)
y = adult_data[target_name]

IndList = range(adult_data.shape[0])
train_list, test_list = train_test_split(IndList, random_state=SEED)
val_list, test_list = train_test_split(
    test_list, random_state=SEED, test_size=0.5)

train = adult_data.iloc[train_list]
valid = adult_data.iloc[val_list]
test = adult_data.iloc[test_list]

lgbtrain = lgbDataset(
    train.drop(target_name, axis=1),
    train[target_name],
    categorical_feature=cat_col_names,
    free_raw_data=False,
)

lgbvalid = lgbDataset(
    valid.drop(target_name, axis=1),
    valid[target_name],
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
params = copy(optimizer.best)
params['n_estimators'] = 1000

flgbtrain = lgbDataset(
    ftrain.drop(target_name, axis=1),
    ftrain[target_name],
    categorical_feature=cat_col_names,
    free_raw_data=False,
)
lgbtest = lgbDataset(
    test.drop(target_name, axis=1),
    test[target_name],
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
auc = roc_auc_score(lgbtest.label, preds)
f1 = f1_score(lgbtest.label, preds)
print(f'Accuracy: {acc}')

# SAVE
suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
results_filename = '_'.join(['adult_lightgbm', suffix]) + '.json'
results_d = {}
results_d['best_params'] = optimizer.best
results_d['runtime'] = runtime
results_d['acc'] = acc
results_d['auc'] = auc
results_d['f1'] = f1
with open(RESULTS_DIR / results_filename, 'w') as f:
    json.dump(results_d, f, indent=4)

model_filename = '_'.join(['model_adult_lightgbm', suffix]) + '.pkl'
with open(MODELS_DIR / model_filename, 'wb') as f:
    pickle.dump(model, f)
