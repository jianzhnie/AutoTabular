import os
from pathlib import Path

import numpy as np
import pandas as pd
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns
from autofe.get_feature import get_cross_columns
from pytorch_widedeep.metrics import R2Score
from pytorch_widedeep.models import FTTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from pytorch_widedeep.training import Trainer

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/house/'
    RESULTS_DIR = ROOTDIR / 'results/house/logistic_regression'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    len_train = len(train_data)

    target_name = 'SalePrice'
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    print(total_data.info())
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    cat_col_names = get_category_columns(total_data, target_name)
    num_col_names = get_numerical_columns(total_data, target_name)
    crossed_cols = get_cross_columns(cat_col_names)

    wide_prprocessor = WidePreprocessor(cat_col_names)
    X_wide = wide_prprocessor.fit_transform(total_data)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=num_col_names,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(total_data)
    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=36)

    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)
    model = WideDeep(wide=wide, deeptabular=ft_transformer)
    trainer = Trainer(model, objective='rmse', metrics=[R2Score])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target_name,
        n_epochs=30,
        batch_size=256,
        val_split=0.2)
