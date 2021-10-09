# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model."""
import logging
from dataclasses import dataclass, field
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabular.config import ModelConfig
from pytorch_tabular.models import BaseModel
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

logger = logging.getLogger(__name__)


class FeedForwardBackbone(pl.LightningModule):

    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        # Linear Layers
        layers = []
        _curr_units = self.embedding_cat_dim + self.hparams.continuous_dim
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
        for units in self.hparams.layers.split('-'):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                ))
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        self.output_dim = _curr_units

    def forward(self, x):
        x = self.linear_layers(x)
        return x


class WideAndDeepBackbone(pl.LightningModule):
    """Standard feedforward layers with a single skip connection from output
    directly to input (ie.

    deep and wide network).
    """

    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        # Linear Layers
        layers = []
        _curr_units = self.embedding_cat_dim + self.hparams.continuous_dim
        self.input_dim = _curr_units
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
        for units in self.hparams.layers.split('-'):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                ))
            _curr_units = int(units)
        self.deep_layers = nn.Sequential(*layers)
        self.output_dim = _curr_units
        self.wide_layers = nn.Linear(
            in_features=self.input_dim, out_features=self.output_dim)

    def forward(self, x):
        x = self.deep_layers(x) + self.wide_layers(x)
        return x


class CategoryEmbeddingModel(BaseModel):

    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims])
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(
                self.hparams.continuous_dim)
        # Backbone
        self.backbone = FeedForwardBackbone(self.hparams)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams.activation,
                           self.hparams.initialization, self.output_layer)

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x['continuous'], x['categorical']
        if self.embedding_cat_dim != 0:
            x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        x = self.backbone(x)
        y_hat = self.output_layer(x)
        if (self.hparams.task == 'regression') and (self.hparams.target_range
                                                    is not None):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (
                    y_max - y_min)
        return {'logits': y_hat, 'backbone_features': x}


class WideDeepEmbeddingModel(BaseModel):

    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims])
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(
                self.hparams.continuous_dim)
        # Backbone
        self.backbone = WideAndDeepBackbone(self.hparams)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams.activation,
                           self.hparams.initialization, self.output_layer)

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x['continuous'], x['categorical']
        if self.embedding_cat_dim != 0:
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        x = self.backbone(x)
        y_hat = self.output_layer(x)
        if (self.hparams.task == 'regression') and (self.hparams.target_range
                                                    is not None):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (
                    y_max - y_min)
        return {'logits': y_hat, 'backbone_features': x}


@dataclass
class WideDeepEmbeddingModelConfig(ModelConfig):
    """CategoryEmbeddingModel configuration
    Args:
        task (str): Specify whether the problem is regression of classification.Choices are: regression classification
        learning_rate (float): The learning rate of the model
        loss (Union[str, NoneType]): The loss function to be applied.
            By Default it is MSELoss for regression and CrossEntropyLoss for classification.
            Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification
        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training.
            The metrics should be one of the metrics implemented in PyTorch Lightning.
            By default, it is Accuracy if classification and MeanSquaredLogError for regression
        metrics_params (Union[List, NoneType]): The parameters to be passed to the Metrics initialized
        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Currently ignored for multi-target regression
            Typically used for Regression problems. If left empty, will not apply any restrictions

        layers (str): Hyphen-separated number of layers and units in the classification head. eg. 32-64-32.
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer
        activation (str): The activation type in the classification head.
            The default activation in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column
            as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column
            using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): probability of an embedding element to be zeroed.
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut
        initialization (str): Initialization scheme for the linear layers. Choices are: `kaiming` `xavier` `random`

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    layers: str = field(
        default='128-64-32',
        metadata={
            'help':
            'Hyphen-separated number of layers and units in the classification head. eg. 32-64-32.'
        },
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={
            'help':
            'If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer'
        },
    )
    activation: str = field(
        default='ReLU',
        metadata={
            'help':
            'The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, \
                LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity'
        },
    )
    embedding_dropout: float = field(
        default=0.5,
        metadata={'help': 'probability of an embedding element to be zeroed.'},
    )
    dropout: float = field(
        default=0.5,
        metadata={
            'help': 'probability of an classification element to be zeroed.'
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={
            'help':
            'Flag to include a BatchNorm layer after each Linear Layer+DropOut'
        },
    )
    initialization: str = field(
        default='kaiming',
        metadata={
            'help': 'Initialization scheme for the linear layers',
            'choices': ['kaiming', 'xavier', 'random'],
        },
    )
    _module_src: str = field(default='category_embedding')
    _model_name: str = field(default='WideDeepEmbeddingModel')
    _config_name: str = field(default='WideDeepEmbeddingModelConfig')
