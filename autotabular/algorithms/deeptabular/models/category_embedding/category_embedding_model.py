'''
Author: jianzhnie
Date: 2021-11-05 16:21:58
LastEditTime: 2021-11-05 16:32:36
LastEditors: jianzhnie
Description: 
'''

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
from pathlib import Path


if __name__ == '__main__':
    BASE_DIR = Path.home().joinpath('data')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    target_name = ["Covertype"]

    cat_col_names = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]

    num_col_names = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (
        num_col_names + cat_col_names + target_name)

    df = pd.read_csv(datafile, header=None, names=feature_columns)
    # cat_col_names = []

    # num_col_names = [
    #     "Elevation", "Aspect"
    # ]
    # feature_columns = (
    #     num_col_names + cat_col_names + target_name)
    # df = df.loc[:,feature_columns]
    print(df.head())
    print(df.shape)
    train, test = train_test_split(df, random_state=42)
    train, val = train_test_split(train, random_state=42)
    num_classes = len(set(train[target_name].values.ravel()))
