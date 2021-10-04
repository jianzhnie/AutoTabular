import os

import pandas as pd
import torch
import torch.nn.functional as F
from autofe.tabular_embedding.pytorch_tabular import FeedForwardNN, TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


def get_optimizer(model, lr=0.001, wd=0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


if __name__ == '__main__':
    data_dir = 'autotabular/datasets/data/shelter-animal-outcomes'
    data_dir = '/media/robin/DATA/datatsets/structure_data/shelter-animal-outcomes'
    data = pd.read_csv(
        os.path.join(data_dir, 'train.csv'),
        usecols=[
            'Name', 'OutcomeType', 'AnimalType', 'SexuponOutcome',
            'AgeuponOutcome', 'Breed', 'Color'
        ])
    categorical_features = [
        'Name', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed',
        'Color'
    ]
    output_feature = 'OutcomeType'
    for col in data.columns:
        if data[col].isnull().sum() > 10000:
            print('dropping', col, data[col].isnull().sum())
            data = data.drop(columns=[col])

    for col in categorical_features:
        if data.dtypes[col] == 'object':
            data[col] = data[col].fillna('NA')
        else:
            data[col] = data[col].fillna(0)
        data[col] = LabelEncoder().fit_transform(data[col])

    for col in categorical_features:
        data[col] = data[col].astype('category')

    cat_dims = [int(data[col].nunique()) for col in categorical_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    data[output_feature] = LabelEncoder().fit_transform(data[output_feature])
    output_size = int(data[output_feature].nunique())
    print(output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardNN(
        emb_dims,
        no_of_cont=0,
        lin_layer_sizes=[50, 100],
        output_size=1,
        emb_dropout=0.4,
        lin_layer_dropouts=[0.2, 0.1]).to(device)

    def train_model(model, optim, train_dl):
        model.train()
        total = 0
        sum_loss = 0
        for y, cont_x, cat_x in train_dl:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            y = y.to(device)
            batch = y.shape[0]
            output = model(cont_x, cat_x)
            loss = F.mse_loss(output, y)
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
        for y, cont_x, cat_x in valid_dl:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            y = y.to(device)
            current_batch_size = y.shape[0]
            out = model(cont_x, cat_x)
            loss = F.mse_loss(out, y)
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

    train_data, val_data = train_test_split(data, shuffle=False)
    batch_size = 128
    train_ds = TabularDataset(
        data=train_data,
        cat_cols=categorical_features,
        output_col=output_feature)

    valid_ds = TabularDataset(
        data=val_data,
        cat_cols=categorical_features,
        output_col=output_feature)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_loop(model, epochs=100, lr=0.05, wd=0.00001)
