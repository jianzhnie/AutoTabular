import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def compute_score(y_true, y_pred, round_digits=3):
    log_loss = round(metrics.log_loss(y_true, y_pred), round_digits)
    auc = round(metrics.roc_auc_score(y_true, y_pred), round_digits)

    precision, recall, threshold = metrics.precision_recall_curve(
        y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)

    mask = ~np.isnan(f1)
    f1 = f1[mask]
    precision = precision[mask]
    recall = recall[mask]

    best_index = np.argmax(f1)
    threshold = round(threshold[best_index], round_digits)
    precision = round(precision[best_index], round_digits)
    recall = round(recall[best_index], round_digits)
    f1 = round(f1[best_index], round_digits)

    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'log_loss': log_loss
    }


def predict(tabular_model, tabular_data_module):
    data_loader = tabular_data_module.test_dataloader()
    batch_size = data_loader.batch_size
    n_rows = len(tabular_data_module.dataset_test)

    y_true = np.zeros(n_rows, dtype=np.float32)
    y_pred = np.zeros(n_rows, dtype=np.float32)
    with torch.no_grad():
        idx = 0
        for num_batch, cat_batch, label_batch in data_loader:
            y_output = tabular_model(num_batch, cat_batch)

            # we convert the output value to binary classification probability
            # with a sigmoid operation, note that this step is specific to the
            # problem at hand, and might not apply to say a regression problem
            y_prob = torch.sigmoid(y_output).cpu().numpy()

            start_idx = idx
            idx += batch_size
            end_idx = idx
            y_pred[start_idx:end_idx] = y_prob
            y_true[start_idx:end_idx] = label_batch.cpu().numpy()

            if end_idx == n_rows:
                break

    return y_true, y_pred


class TabularDataset(Dataset):

    def __init__(self, path, num_cols, cat_cols, label_col):
        self.path = path
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.df = self.read_data(path, num_cols, cat_cols, label_col)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        num_array = self.df[self.num_cols].iloc[idx].values
        cat_array = self.df[self.cat_cols].iloc[idx].values
        label_array = self.df[self.label_col].iloc[idx]
        return num_array, cat_array, label_array

    def read_data(self, path, num_cols, cat_cols, label_col):
        float_cols = num_cols + [label_col]
        dtype = {col: np.float32 for col in float_cols}
        dtype.update({col: np.int64 for col in cat_cols})
        return pd.read_csv(path, dtype=dtype)


class TabularDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 num_cols,
                 cat_cols,
                 label_col,
                 num_workers=4,
                 batch_size_train=128,
                 batch_size_val=64,
                 batch_size_test=512):
        super().__init__()
        self.data_dir = data_dir
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.num_workers = num_workers
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

    def setup(self, stage):
        num_cols = self.num_cols
        cat_cols = self.cat_cols
        label_col = self.label_col

        path_train = os.path.join(self.data_dir, 'train.csv')
        self.dataset_train = TabularDataset(path_train, num_cols, cat_cols,
                                            label_col)

        path_val = os.path.join(self.data_dir, 'val.csv')
        self.dataset_val = TabularDataset(path_val, num_cols, cat_cols,
                                          label_col)

        path_test = os.path.join(self.data_dir, 'test.csv')
        self.dataset_test = TabularDataset(path_test, num_cols, cat_cols,
                                           label_col)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size_train,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size_val,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size_test,
            shuffle=False)


class TabularNet(pl.LightningModule):

    def __init__(self,
                 num_cols,
                 cat_cols,
                 embedding_size_dict,
                 n_classes,
                 embedding_dim_dict=None,
                 learning_rate=0.01):
        super().__init__()

        # pytorch lightning black magic, all the arguments can now be
        # accessed through self.hparams.[argument]
        self.save_hyperparameters()

        self.embeddings, total_embedding_dim = self._create_embedding_layers(
            cat_cols, embedding_size_dict, embedding_dim_dict)

        # concatenate the numerical variables and the embedding layers
        # then proceed with the rest of the sequential flow
        in_features = len(num_cols) + total_embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 256),
            nn.ReLU(), nn.Linear(256, n_classes))

    @staticmethod
    def _create_embedding_layers(cat_cols, embedding_size_dict,
                                 embedding_dim_dict):
        """construct the embedding layer, 1 per each categorical variable."""
        total_embedding_dim = 0
        embeddings = {}
        for col in cat_cols:
            embedding_size = embedding_size_dict[col]
            embedding_dim = embedding_dim_dict[col]
            total_embedding_dim += embedding_dim
            embeddings[col] = nn.Embedding(embedding_size, embedding_dim)

        return nn.ModuleDict(embeddings), total_embedding_dim

    def forward(self, num_tensor, cat_tensor):

        # run through all the categorical variables through its
        # own embedding layer and concatenate them together
        cat_outputs = []
        for i, col in enumerate(self.hparams.cat_cols):
            embedding = self.embeddings[col]
            cat_output = embedding(cat_tensor[:, i])
            cat_outputs.append(cat_output)

        cat_outputs = torch.cat(cat_outputs, dim=1)

        # concatenate the categorical embedding and numerical layer
        all_outputs = torch.cat((num_tensor, cat_outputs), dim=1)

        # for binary classification or regression we don't need the additional dimension
        final_outputs = self.layers(all_outputs).squeeze(dim=1)
        return final_outputs

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)

    def compute_loss(self, batch, batch_idx):
        num_tensor, cat_tensor, label_tensor = batch
        output_tensor = self(num_tensor, cat_tensor)
        loss = F.binary_cross_entropy_with_logits(output_tensor, label_tensor)
        return loss


def emb_sz_rule(n_cat):
    """Rule of thumb to pick embedding size corresponding to `n_cat`"""
    return min(600, round(1.6 * n_cat**0.56))
