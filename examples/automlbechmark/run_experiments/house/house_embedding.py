from pathlib import Path

import pandas as pd
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns
from pytorch_tabular.config import DataConfig, ExperimentConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.tabular_model import TabularModel

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/house/'

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    len_train = len(train_data)

    target_name = 'SalePrice'

    cat_col_names = get_category_columns(total_data, target_name)
    num_col_names = get_numerical_columns(total_data, target_name)
    date_columns = []

    data_config = DataConfig(
        target=target_name,
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
        date_columns=date_columns,
        normalize_continuous_features=True,
    )

    tab_transformer_config = TabTransformerConfig(
        task='regression',
        metrics=['r2'],
        share_embedding=True,
        share_embedding_strategy='add',
        shared_embedding_fraction=0.25)

    trainer_config = TrainerConfig(
        gpus=1,
        auto_lr_find=True,
        auto_select_gpus=True,
        max_epochs=1,
        batch_size=1024)

    experiment_config = ExperimentConfig(
        project_name='PyTorch Tabular Example',
        run_name='node_forest_cov',
        exp_watch='gradients',
        log_logits=True)

    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=tab_transformer_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config)

    tabular_model.fit(train=total_data)
    pred_df = tabular_model.predict(test_data, ret_logits=True)
    print(pred_df)
