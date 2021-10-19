import json
import os
from datetime import datetime
from pathlib import Path

import flash
import numpy as np
import pandas as pd
import torch
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from flash.tabular import TabularClassificationData, TabularClassifier
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.utils import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

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

    # TRAIN/VALID for hyperparam optimization
    label_encoder = LabelEncoder(cat_col_names)
    bankm_train = label_encoder.fit_transform(bankm_train)
    bankm_test = label_encoder.transform(bankm_test)

    X_train = bankm_train.drop(target_name, axis=1)
    y_train = bankm_train[target_name]

    X_test = bankm_test.drop(target_name, axis=1)
    y_test = bankm_test[target_name]

    clf = LogisticRegression(max_iter=1000, warm_start=True, tol=1e-4)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(clf)
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')

    # SAVE
    base_lr = {}
    base_lr['acc'] = acc
    base_lr['auc'] = auc
    base_lr['f1'] = f1

    # random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(clf)
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    base_randomforest = {}
    base_randomforest['acc'] = acc
    base_randomforest['auc'] = auc
    base_randomforest['f1'] = f1

    # xgboost
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    print(clf)
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    # SAVE
    base_xgboost = {}
    base_xgboost['acc'] = acc
    base_xgboost['auc'] = auc
    base_xgboost['f1'] = f1

    # gbdt feature
    lgb = LightGBMFeatureTransformer(
        task='classification',
        categorical_feature=cat_col_names,
        params={
            'n_estimators': 100,
            'max_depth': 3
        })
    lgb.fit(X_train, y_train)
    X_enc_train = lgb.concate_transform(X_train, concate=False)
    X_enc_test = lgb.concate_transform(X_test, concate=False)

    clf = LogisticRegression(
        max_iter=int(1e4), warm_start=True, tol=1e-4, penalty='l2')

    clf.fit(X_enc_train, y_train)
    preds = clf.predict(X_enc_test)
    preds_prob = clf.predict_proba(X_enc_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print('GBDT + lr')
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    gbdt_lr = {}
    gbdt_lr['acc'] = acc
    gbdt_lr['auc'] = auc
    gbdt_lr['f1'] = f1

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

    # embedding
    target = bankm_train[target_name].values
    wide_cols = cat_col_names
    crossed_cols = []
    for i in range(0, len(wide_cols) - 1):
        for j in range(i + 1, len(wide_cols)):
            crossed_cols.append((wide_cols[i], wide_cols[j]))

    wide_prprocessor = WidePreprocessor(wide_cols, crossed_cols)
    X_wide = wide_prprocessor.fit_transform(bankm_train)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=num_cols,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(bankm_train)

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=36)

    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)
    model = WideDeep(wide=wide, deeptabular=ft_transformer)
    trainer = Trainer(model, objective='binary', metrics=[Accuracy])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=30,
        batch_size=512,
        val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec_train = t2v.transform(bankm_train)
    X_vec_test = t2v.transform(bankm_test)
    feature_names = ['nn_embed_' + str(i) for i in range(X_vec_train.shape[1])]
    X_vec_train = pd.DataFrame(X_vec_train, columns=feature_names)
    X_vec_test = pd.DataFrame(X_vec_test, columns=feature_names)

    clf = LogisticRegression(
        max_iter=int(1e4), warm_start=True, tol=1e-4, penalty='l2')
    clf.fit(X_vec_train, y_train)
    preds = clf.predict(X_vec_test)
    preds_prob = clf.predict_proba(X_vec_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)

    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    # SAVE

    suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
    results_filename = '_'.join(['bankm_results', suffix]) + '.json'
    nn_embedding_lr = {}
    nn_embedding_lr['acc'] = acc
    nn_embedding_lr['auc'] = auc
    nn_embedding_lr['f1'] = f1
    results = {
        'nn_embedding_lr': nn_embedding_lr,
        'gbdt_lr': gbdt_lr,
        'base_lr': base_lr
    }
    with open(RESULTS_DIR / results_filename, 'w') as f:
        json.dump(results, f, indent=4)
