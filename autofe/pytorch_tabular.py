import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):

    def __init__(self, data, cat_cols=None, output_col=None):
        """Characterizes a Dataset for PyTorch.

        Parameters
        ----------

        data: pandas data frame
        The data frame object for the input data. It must
        contain all the continuous, categorical and the
        output columns to be used.

        cat_cols: List of strings
        The names of the categorical columns in the data.
        These columns will be passed through the embedding
        layers in the model. These columns must be
        label encoded beforehand.

        output_col: string
        The name of the output variable column in the data
        provided.
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [
            col for col in data.columns
            if col not in self.cat_cols + [output_col]
        ]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """Denotes the total number of samples."""
        return self.n

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


class FeedForwardNN(nn.Module):

    def __init__(
        self,
        emb_dims,
        no_of_cont,
        lin_layer_sizes,
        output_size,
        emb_dropout,
        lin_layer_dropouts,
    ):
        """
        Parameters
        ----------

        emb_dims: List of two element tuples
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.

        no_of_cont: Integer
        The number of continuous features in the data.

        lin_layer_sizes: List of integers.
        The size of each linear layer. The length will be equal
        to the total number
        of linear layers in the network.

        output_size: Integer
        The size of the final output.

        emb_dropout: Float
        The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
        The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [
            nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
            for i in range(len(lin_layer_sizes) - 1)
        ])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [
                emb_layer(cat_data[:, i])
                for i, emb_layer in enumerate(self.emb_layers)
            ]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers,
                                                      self.droput_layers,
                                                      self.bn_layers):

            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x


if __name__ == '__main__':
    # data url
    """https://www.kaggle.com/c/house-prices-advanced-regression-techniques."""
    # Using only a subset of the variables.
    data_dir = '/media/robin/DATA/datatsets/structure_data/house_price/train.csv'
    data = pd.read_csv(
        data_dir,
        usecols=[
            'SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
            'Street', 'YearBuilt', 'LotShape', '1stFlrSF', '2ndFlrSF'
        ]).dropna()

    categorical_features = [
        'MSSubClass', 'MSZoning', 'Street', 'LotShape', 'YearBuilt'
    ]
    output_feature = 'SalePrice'
    label_encoders = {}
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

    dataset = TabularDataset(
        data=data, cat_cols=categorical_features, output_col=output_feature)

    batchsize = 64
    dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

    cat_dims = [int(data[col].nunique()) for col in categorical_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardNN(
        emb_dims,
        no_of_cont=4,
        lin_layer_sizes=[50, 100],
        output_size=1,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]).to(device)
    print(model)
    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(num_epochs):
        for y, cont_x, cat_x in dataloader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            y = y.to(device)
            # Forward Pass
            preds = model(cont_x, cat_x)
            loss = criterion(preds, y)
            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss:', loss)
