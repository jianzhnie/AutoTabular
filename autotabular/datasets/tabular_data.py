# Pytorch Tabular
"""Tabular Data Module"""
from torch.utils.data import Dataset
from typing import List
import numpy as np
import pandas as pd


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        embed_categorical: bool = True,
        target: List[str] = None,
    ):
        """Dataset to Load Tabular Data

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training
            task (str): Whether it is a classification or regression task. If classification, it returns a LongTensor as target
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
            These columns must be ordinal encoded beforehand. Defaults to None.
            embed_categorical (bool): Flag to tell the dataset whether to convert categorical columns to LongTensor or retain as float.
            If we are going to embed categorical cols with an embedding layer, we need to convert the columns to LongTensor
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.
        """

        self.task = task
        self.n = data.shape[0]

        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)  # .astype(np.int64)
        else:
            self.y = np.zeros((self.n, 1))  # .astype(np.int64)

        if task == "classification":
            self.y = self.y.astype(np.int64)
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(
                np.float32).values

        if self.categorical_cols:
            self.categorical_X = data[categorical_cols]
            if embed_categorical:
                self.categorical_X = self.categorical_X.astype(np.int64).values
            else:
                self.categorical_X = self.categorical_X.astype(
                    np.float32).values

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return {
            "target":
            self.y[idx],
            "continuous":
            self.continuous_X[idx] if self.continuous_cols else [],
            "categorical":
            self.categorical_X[idx] if self.categorical_cols else [],
        }
