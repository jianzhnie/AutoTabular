import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/bank_marketing/'

    RESULTS_DIR = ROOTDIR / 'results/bank_marketing/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    bank_maket = pd.read_csv(PROCESSED_DATA_DIR / 'bankm.csv')
    target_name = 'target'

    IndList = range(bank_maket.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = bank_maket.iloc[train_list]
    val = bank_maket.iloc[val_list]
    test = bank_maket.iloc[test_list]

    # network transformer
    cat_col_names, cont_col_names = [], []
    for col in bank_maket.columns:
        if bank_maket[col].dtype == 'O' and col != 'target':
            cat_col_names.append(col)
        elif bank_maket[col].dtype in ['float', 'int'] and col != 'target':
            cont_col_names.append(col)

    print(bank_maket.info())
    print(cat_col_names, cont_col_names)

    X = bank_maket.drop(target_name, axis=1)
    y = bank_maket[target_name]
    target = bank_maket[target_name].values

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=cont_col_names,
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
        X_tab=X_tab, target=target, n_epochs=30, batch_size=256, val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec = t2v.transform(bank_maket)
    feature_names = ['nn_embed_' + str(i) for i in range(X_vec.shape[1])]
    X_vec = pd.DataFrame(X_vec, columns=feature_names)

    selector = SelectFromModel(
        estimator=LogisticRegression(), max_features=64).fit(X_vec, y)
    support = selector.get_support()
    col_names = X_vec.columns[support]
    X_enc = selector.transform(X_vec)
    X_enc = pd.DataFrame(X_enc, columns=col_names)

    X_enc = pd.concat([X, X_enc])
    X_enc[target_name] = y
    train_enc = X_enc.iloc[train_list]
    val_enc = X_enc.iloc[val_list]
    test_enc = X_enc.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, eval_metric='roc_auc', path=RESULTS_DIR).fit(
            train_data=train_enc, tuning_data=val_enc)

    scores = predictor.evaluate(test_enc, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(test_enc)
