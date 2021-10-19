import os
from pathlib import Path

import flash
import numpy as np
import pandas as pd
import torch
from flash.tabular import TabularClassificationData, TabularClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42


def generate_cross_cols(self, df: pd.DataFrame, crossed_cols):
    df_cc = df.copy()
    crossed_colnames = []
    for cols in crossed_cols:
        for c in cols:
            df_cc[c] = df_cc[c].astype('str')
        colname = '_'.join(cols)
        df_cc[colname] = df_cc[list(cols)].apply(lambda x: '-'.join(x), axis=1)

        crossed_colnames.append(colname)
    return df_cc[crossed_colnames]


if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/bank_marketing/'
    RESULTS_DIR = ROOTDIR / 'results/bank_marketing/logistic_regression'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    bankm_train = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    bankm_test = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')

    target_name = 'target'
    print(bankm_train.info())
    cat_col_names = []
    for col in bankm_train.columns:
        if bankm_train[col].dtype == 'object' and col != 'target':
            cat_col_names.append(col)

    num_cols = [
        c for c in bankm_train.columns if c not in cat_col_names + ['target']
    ]

    X_train = bankm_train.drop(target_name, axis=1)
    y_train = bankm_train[target_name]

    X_test = bankm_test.drop(target_name, axis=1)
    y_test = bankm_test[target_name]

    # tabnet
    # 1. Create the DataModule
    datamodule = TabularClassificationData.from_data_frame(
        categorical_fields=cat_col_names,
        numerical_fields=num_cols,
        target_fields=target_name,
        train_data_frame=bankm_train,
        val_data_frame=bankm_test,
        batch_size=128,
    )
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
