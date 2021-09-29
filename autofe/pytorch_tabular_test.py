import pandas as pd
import torch
import torch.nn as nn
from autofe.tabular_embedding.pytorch_tabular import FeedForwardNN, TabularDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

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
