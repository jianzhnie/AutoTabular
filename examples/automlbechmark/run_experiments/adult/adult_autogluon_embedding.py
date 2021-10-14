import os
from pathlib import Path

import pandas as pd
from autofe.tabular_embedding.tabular_embedding_transformer import TabularEmbeddingTransformer
from autogluon.tabular import TabularPredictor
from pytorch_tabular.config import DataConfig, ExperimentConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.tabular_model import TabularModel
from pytorch_widedeep.utils import LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'
    init_args = {'eval_metric': 'roc_auc', 'path': RESULTS_DIR}

    cat_col_names = []
    for col in adult_data.columns:
        if adult_data[col].dtype == 'object' and col != 'target':
            cat_col_names.append(col)

    num_col_names = []
    for col in adult_data.columns:
        if adult_data[col].dtype == 'float' and col != 'target':
            num_col_names.append(col)
    date_columns = []

    num_classes = len(set(adult_data[target_name].values.ravel()))

    print(num_classes)
    print(cat_col_names)
    print(num_col_names)
    print(adult_data.info())
    print(adult_data.describe())

    label_encoder = LabelEncoder(cat_col_names)
    adult_data = label_encoder.fit_transform(adult_data)

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

    data_config = DataConfig(
        target=target_name,
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
        date_columns=date_columns,
        continuous_feature_transform='quantile_normal',
        normalize_continuous_features=True,
    )

    tab_transformer_config = TabTransformerConfig(
        task='classification',
        metrics=['f1', 'accuracy'],
        share_embedding=True,
        share_embedding_strategy='add',
        shared_embedding_fraction=0.25,
        metrics_params=[{
            'num_classes': num_classes,
            'average': 'macro'
        }, {}],
    )

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

    tabular_model.fit(train=adult_data)
    dt = DeepFeatureExtractor(tabular_model)
    X_enc = dt.fit_transform(adult_data)
    train_enc = X_enc.iloc[train_list]
    val_enc = X_enc.iloc[val_list]
    test_enc = X_enc.iloc[test_list]

    predictor = TabularPredictor(label=target_name).fit(
        train_data=train_enc, tuning_data=val_enc)

    scores = predictor.evaluate(test_enc, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test_enc)

    transformer = TabularEmbeddingTransformer(
        cat_col_names=cat_col_names,
        num_col_names=num_col_names,
        date_col_names=[],
        target_name=target_name,
        num_classes=num_classes)

    print(transformer)
    X_enc = transformer.fit_transform(adult_data)
    train_enc = X_enc.iloc[train_list]
    val_enc = X_enc.iloc[val_list]
    test_enc = X_enc.iloc[test_list]

    predictor = TabularPredictor(label=target_name).fit(
        train_data=train_enc, tuning_data=val_enc)

    scores = predictor.evaluate(test_enc, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test_enc)
