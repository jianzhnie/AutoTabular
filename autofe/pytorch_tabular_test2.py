import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from autofe.tabular_embedding.pytorch_tabular_ import DeviceDataLoader, ShelterOutcomeDataset, ShelterOutcomeModel, get_default_device, get_optimizer, to_device
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

if __name__ == '__main__':
    data_dir = 'autotabular/datasets/data/shelter-animal-outcomes'
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    print('Shape:', train.shape)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print('Shape:', test.shape)
    train_X = train.drop(columns=['OutcomeType', 'OutcomeSubtype', 'AnimalID'])
    Y = train['OutcomeType']
    test_X = test
    stacked_df = train_X.append(test_X.drop(columns=['ID']))
    stacked_df = stacked_df.drop(columns=['DateTime'])
    for col in stacked_df.columns:
        if stacked_df[col].isnull().sum() > 10000:
            print('dropping', col, stacked_df[col].isnull().sum())
            stacked_df = stacked_df.drop(columns=[col])

    for col in stacked_df.columns:
        if stacked_df.dtypes[col] == 'object':
            stacked_df[col] = stacked_df[col].fillna('NA')
        else:
            stacked_df[col] = stacked_df[col].fillna(0)
        stacked_df[col] = LabelEncoder().fit_transform(stacked_df[col])

    for col in stacked_df.columns:
        stacked_df[col] = stacked_df[col].astype('category')

    X = stacked_df[0:26729]
    test_processed = stacked_df[26729:]

    Y = LabelEncoder().fit_transform(Y)
    print(Counter(train['OutcomeType']))
    print(Counter(Y))
    target_dict = {
        'Return_to_owner': 3,
        'Euthanasia': 2,
        'Adoption': 0,
        'Transfer': 4,
        'Died': 1
    }
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.10, random_state=0)
    X_train.head()
    embedded_cols = {
        n: len(col.cat.categories)
        for n, col in X.items() if len(col.cat.categories) > 2
    }
    embedded_cols

    embedded_col_names = embedded_cols.keys()
    len(X.columns) - len(embedded_cols)  # number of numerical columns

    embedding_sizes = [(n_categories, min(50, (n_categories + 1) // 2))
                       for _, n_categories in embedded_cols.items()]

    train_ds = ShelterOutcomeDataset(X_train, y_train, embedded_col_names)
    valid_ds = ShelterOutcomeDataset(X_val, y_val, embedded_col_names)
    device = get_default_device()
    model = ShelterOutcomeModel(embedding_sizes, 1)
    to_device(model, device)

    def train_model(model, optim, train_dl):
        model.train()
        total = 0
        sum_loss = 0
        for x1, x2, y in train_dl:
            batch = y.shape[0]
            output = model(x1, x2)
            print(output)
            print(y)
            loss = F.cross_entropy(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())
        return sum_loss / total

    def val_loss(model, valid_dl):
        model.eval()
        total = 0
        sum_loss = 0
        correct = 0
        for x1, x2, y in valid_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)
            loss = F.cross_entropy(out, y)
            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
            pred = torch.max(out, 1)[1]
            correct += (pred == y).float().sum().item()
        print('valid loss %.3f and accuracy %.3f' %
              (sum_loss / total, correct / total))
        return sum_loss / total, correct / total

    def train_loop(model, epochs, lr=0.01, wd=0.0):
        optim = get_optimizer(model, lr=lr, wd=wd)
        for i in range(epochs):
            loss = train_model(model, optim, train_dl)
            print('training loss: ', loss)
            val_loss(model, valid_dl)

    batch_size = 1000
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    train_loop(model, epochs=8, lr=0.05, wd=0.00001)

    test_ds = ShelterOutcomeDataset(test_processed,
                                    np.zeros(len(test_processed)),
                                    embedded_col_names)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for x1, x2, y in test_dl:
            out = model(x1, x2)
            prob = F.softmax(out, dim=1)
            preds.append(prob)
