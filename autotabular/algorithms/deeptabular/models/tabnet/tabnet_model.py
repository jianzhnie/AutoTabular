'''
Author: jianzhnie
Date: 2021-11-05 17:48:04
LastEditTime: 2021-11-05 17:59:47
LastEditors: jianzhnie
Description: 
'''

import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabnet.tab_network import TabNet

logger = logging.getLogger(__name__)


class TabNetBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        self.tabnet = TabNet(
            input_dim = self.config.input_dim,
            output_dim = self.config.output_dim,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="sparsemax",
        )

    def forward(self, x: Dict):
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        return x


class TabNetModel(pl.LightningDataModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self.backbone = TabNetBackbone(self.hparams)

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = self.unpack_input(x)
        # Returns output
        x = self.backbone(x)
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return {"logits": x}  # No Easy way to access the raw features in TabNet
