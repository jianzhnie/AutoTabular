import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.utils import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42

pd.options.display.max_columns = 100

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/bank_marketing/'
    RESULTS_DIR = ROOTDIR / 'results/bank_marketing/logistic_regression'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    bank_maket = pd.read_csv(PROCESSED_DATA_DIR / 'bankm.csv')
    target_name = 'target'

    print(bank_maket.info())
    cat_col_names = []
    for col in bank_maket.columns:
        if bank_maket[col].dtype == 'O' and col != 'target':
            cat_col_names.append(col)

    num_cols = [
        c for c in bank_maket.columns if c not in cat_col_names + ['target']
    ]

    #  TRAIN/VALID for hyperparam optimization
    label_encoder = LabelEncoder(cat_col_names)
    bank_maket = label_encoder.fit_transform(bank_maket)

    X = bank_maket.drop(target_name, axis=1)
    y = bank_maket[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=SEED)

    clf = LogisticRegression(
        max_iter=int(1e6), warm_start=True, tol=1e-4, penalty='l2')

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    # SAVE
    base_lr = {}
    base_lr['acc'] = acc
    base_lr['auc'] = auc
    base_lr['f1'] = f1

    # gbdt feature
    lgb = LightGBMFeatureTransformer(
        task='classification',
        categorical_feature=cat_col_names,
        params={
            'n_estimators': 100,
            'max_depth': 3
        })
    lgb.fit(X, y)
    X_enc = lgb.concate_transform(X, concate=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, random_state=SEED)

    clf = LogisticRegression(
        max_iter=int(1e4), warm_start=True, tol=1e-4, penalty='l2')

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)

    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    gbdt_lr = {}
    gbdt_lr['acc'] = acc
    gbdt_lr['auc'] = auc
    gbdt_lr['f1'] = f1

    # embedding
    target = bank_maket[target_name].values
    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=num_cols,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(bank_maket)

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=36)

    model = WideDeep(deeptabular=ft_transformer)
    trainer = Trainer(model, objective='binary', metrics=[Accuracy])
    trainer.fit(
        X_tab=X_tab, target=target, n_epochs=30, batch_size=512, val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec = t2v.transform(bank_maket)
    feature_names = ['nn_embed_' + str(i) for i in range(X_vec.shape[1])]
    X_vec = pd.DataFrame(X_vec, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, random_state=SEED)

    clf = LogisticRegression(
        max_iter=int(1e4), warm_start=True, tol=1e-4, penalty='l2')

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)

    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
    # SAVE
    suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
    results_filename = '_'.join(['bankm_ebm_lr', suffix]) + '.json'
    nn_embedding_lr = {}
    nn_embedding_lr['acc'] = acc
    nn_embedding_lr['auc'] = auc
    nn_embedding_lr['f1'] = f1
    results = [nn_embedding_lr, gbdt_lr, base_lr]
    with open(RESULTS_DIR / results_filename, 'w') as f:
        json.dump(results, f, indent=4)
