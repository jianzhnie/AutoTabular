import os
from pathlib import Path

import flash
import numpy as np
import pandas as pd
import torch
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns
from flash.tabular import TabularClassificationData, TabularClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/credit/'
    RESULTS_DIR = ROOTDIR / 'results/credit/logistic_regression'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    len_train = len(train_data)
    print(train_data)
    cat_cols = ['EDUCATION', 'SEX', 'MARRIAGE']
    for c in cat_cols:
        train_data[c] = train_data[c].apply(str)
        test_data[c] = test_data[c].apply(str)

    target_name = 'payment'
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    print(total_data.info())
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    cat_col_names = get_category_columns(total_data, target_name)
    num_col_names = get_numerical_columns(total_data, target_name)

    # tabnet
    # 1. Create the DataModule
    datamodule = TabularClassificationData.from_data_frame(
        categorical_fields=cat_col_names,
        numerical_fields=num_col_names,
        target_fields=target_name,
        train_data_frame=train_data,
        val_data_frame=test_data,
        batch_size=128)
    # 2. Build the task
    model = TabularClassifier.from_data(datamodule)
    # 3. Create the trainer and train the model
    trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
    trainer.fit(model, datamodule=datamodule)
    # 4. Generate predictions from a CSV
    preds_mat = model.predict(test_datafile)
    preds_mat = np.array(preds_mat)
    preds_prob = preds_mat[:, 1]
    print(preds_mat.shape)
    preds = np.argmax(preds_mat, axis=1)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(type(preds))
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
