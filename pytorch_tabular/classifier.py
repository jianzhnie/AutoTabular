"""NeuralNet subclasses for classification tasks."""

import re

import numpy as np
from sklearn.base import ClassifierMixin
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
sys.path.append("../")
from autotabular.deepctr.models.lr import LogisticRegressionModel
from autotabular.datasets.tabular_data import SklearnDataModule
from autotabular.datasets.dsutils import load_heart_disease_uci
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class LR(pl.LightningModule):
    def __init__(self):

        self.model = LogisticRegressionModel(field_dims=[1, 2, 3, 4])

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [checkpoint]


from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
loaders = SklearnDataModule(X, y, batch_size=32)
trainer = pl.Trainer()
model = LR()
trainer.fit(model, datamodule=loaders)
