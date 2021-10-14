import os
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

if __name__ == '__main__':
    pd.options.display.max_columns = 100

    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'

    # 200 is rather arbitraty but one has to make a decision as to how to decide
    # if something will be represented as embeddings or continuous in a "kind-of"
    # automated way
    cat_col_names, cont_col_names = [], []
    for col in adult_data.columns:
        # 50 is just a random number I choose here for this example
        if adult_data[col].dtype == 'O' and col != 'target':
            cat_col_names.append(col)
        elif col != 'target':
            cont_col_names.append(col)

    target = adult_data[target_name].values

    X = adult_data.drop(target_name, axis=1)
    y = adult_data[target_name]

    IndList = range(X.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = adult_data.iloc[train_list]
    val = adult_data.iloc[val_list]
    test = adult_data.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, path=RESULTS_DIR).fit(
            train_data=train, tuning_data=val)

    scores = predictor.evaluate(test, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test)

    wide_cols = [
        'education', 'relationship', 'workclass', 'occupation',
        'native_country', 'sex', 'race', 'marital_status'
    ]

    crossed_cols = [('education', 'occupation'),
                    ('native_country', 'occupation'), ('education', 'sex'),
                    ('education', 'native_country'),
                    ('native_country', 'race')]
    wide_prprocessor = WidePreprocessor(wide_cols, crossed_cols)
    X_wide = wide_prprocessor.fit_transform(adult_data)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=cont_col_names,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(adult_data)

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
        batch_size=256,
        val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec = t2v.transform(adult_data)
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

    predictor = TabularPredictor(label=target_name).fit(
        train_data=train_enc, tuning_data=val_enc)

    scores = predictor.evaluate(test_enc, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test_enc)
