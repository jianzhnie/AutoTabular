from pathlib import Path

import pandas as pd
import wget
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models.category_embedding.config import CategoryEmbeddingModelConfig
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.utils import get_balanced_sampler
from sklearn.model_selection import train_test_split


class TabularEmbeddingTransformer():

    def __init__(self,
                 cat_col_names=None,
                 num_col_names=None,
                 date_col_names=None,
                 target_name=None,
                 num_classes=None):

        self.cat_col_names = cat_col_names
        self.num_col_names = num_col_names
        self.date_col_names = date_col_names
        self.target_name = target_name

        data_config = DataConfig(
            target=target_name,
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
            date_columns=date_col_names,
            continuous_feature_transform='quantile_normal',
            normalize_continuous_features=True,
        )
        model_config = CategoryEmbeddingModelConfig(
            task='classification',
            learning_rate=1e-3,
            metrics=['f1', 'accuracy'],
            metrics_params=[{
                'num_classes': num_classes
            }, {}])

        model_config = TabTransformerConfig(
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
            gpus=-1,
            auto_select_gpus=True,
            auto_lr_find=True,
            max_epochs=120,
            batch_size=1024)

        optimizer_config = OptimizerConfig()

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

    def fit(self, train_data):
        """Just for compatibility.

        Does not do anything
        """
        sampler = get_balanced_sampler(
            train_data[self.target_name].values.ravel())
        self.tabular_model.fit(train=train_data, train_sampler=sampler)
        self.transformerMoldel = DeepFeatureExtractor(
            self.tabular_model, drop_original=False)
        return self

    def transform(self, train_data):
        encoded_data = self.transformerMoldel.transform(train_data)
        return encoded_data

    def fit_transform(self, train_data):
        """Encode given columns of X based on the learned features.

        Args:
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
            y ([type], optional): Only for compatibility. Not used. Defaults to None.

        Returns:
            pd.DataFrame: The encoded dataframe
        """
        self.fit(train_data)
        return self.transform(train_data)


if __name__ == '__main__':
    BASE_DIR = Path.home().joinpath('data')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    if not datafile.exists():
        wget.download(url, datafile.as_posix())
    target_name = ['Covertype']
    cat_col_names = [
        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
    ]

    num_col_names = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]

    date_col_names = []
    feature_columns = (
        num_col_names + cat_col_names + date_col_names + target_name)

    df = pd.read_csv(datafile, header=None, names=feature_columns)

    train, test = train_test_split(df, random_state=42)
    train, val = train_test_split(train, random_state=42)
    num_classes = len(set(train[target_name].values.ravel()))

    # data_config = DataConfig(
    #     target=target_name,
    #     continuous_cols=num_col_names,
    #     categorical_cols=cat_col_names,
    #     continuous_feature_transform=None,  #"quantile_normal",
    #     normalize_continuous_features=False,
    # )
    # model_config = CategoryEmbeddingModelConfig(
    #     task='classification',
    #     metrics=['f1', 'accuracy'],
    #     metrics_params=[{
    #         'num_classes': num_classes
    #     }, {}])
    # model_config = NodeConfig(
    #     task='classification',
    #     depth=4,
    #     num_trees=1024,
    #     input_dropout=0.0,
    #     metrics=['f1', 'accuracy'],
    #     metrics_params=[{
    #         'num_classes': num_classes,
    #         'average': 'macro'
    #     }, {}],
    # )
    # model_config = TabTransformerConfig(
    #     task='classification',
    #     metrics=['f1', 'accuracy'],
    #     share_embedding=True,
    #     share_embedding_strategy='add',
    #     shared_embedding_fraction=0.25,
    #     metrics_params=[{
    #         'num_classes': num_classes,
    #         'average': 'macro'
    #     }, {}],
    # )

    # trainer_config = TrainerConfig(
    #     gpus=-1,
    #     auto_select_gpus=True,
    #     fast_dev_run=True,
    #     max_epochs=5,
    #     batch_size=512)
    # experiment_config = ExperimentConfig(
    #     project_name='PyTorch Tabular Example',
    #     run_name='node_forest_cov',
    #     exp_watch='gradients',
    #     log_target='wandb',
    #     log_logits=True)
    # optimizer_config = OptimizerConfig()
    # tabular_model = TabularModel(
    #     data_config=data_config,
    #     model_config=model_config,
    #     optimizer_config=optimizer_config,
    #     trainer_config=trainer_config,
    #     experiment_config=experiment_config,
    # )
    # sampler = get_balanced_sampler(train[target_name].values.ravel())
    # tabular_model.fit(train=train, validation=val, train_sampler=sampler)

    transformer = TabularEmbeddingTransformer(
        cat_col_names=cat_col_names,
        num_col_names=num_col_names,
        date_columns=[],
        target_name=target_name,
        num_classes=num_classes)
    train_transform = transformer.fit_transform(df)
    print(train_transform)
