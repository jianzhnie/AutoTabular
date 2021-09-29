import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from autofe.tabular_embedding.pytorchlightning_tabular import TabularDataModule, TabularDataset, TabularNet, compute_score, predict
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

if __name__ == '__main__':
    """http://ethen8181.github.io/machine-
    learning/deep_learning/tabular/tabular.html#Deep-Learning-For-Tabular-
    Data."""
    input_path = '/media/robin/DATA/datatsets/structure_data/UCI_Credit_Card/UCI_Credit_Card.csv'
    df = pd.read_csv(input_path)
    print(df.shape)
    print(df.head())
    id_cols = ['ID']
    cat_cols = ['EDUCATION', 'SEX', 'MARRIAGE']
    num_cols = [
        'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
        'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
        'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    label_col = 'default.payment.next.month'

    print('number of categorical columns: ', len(cat_cols))
    print('number of numerical columns: ', len(num_cols))
    test_size = 0.1
    val_size = 0.3
    random_state = 1234
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col])

    df_train, df_val = train_test_split(
        df_train,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train[label_col])

    print('train shape: ', df_train.shape)
    print('validation shape: ', df_val.shape)
    print('test shape: ', df_test.shape)

    cat_code_dict = {}
    for col in cat_cols:
        category_col = df_train[col].astype('category')
        cat_code_dict[col] = {
            value: idx
            for idx, value in enumerate(category_col.cat.categories)
        }
    print(cat_code_dict)

    def preprocess(df,
                   scaler=None,
                   num_cols=None,
                   cat_cols=None,
                   label_col=None):
        df = df.copy()
        # numeric fields
        scaler = StandardScaler()
        scaler.fit(df_train[num_cols])
        df[num_cols] = scaler.transform(df[num_cols])
        df[num_cols] = df[num_cols].astype(np.float32)
        # categorical fields
        # store the category code mapping, so we can encode any new incoming data
        # other than our training set
        cat_code_dict = {}
        for col in cat_cols:
            category_col = df_train[col].astype('category')
            cat_code_dict[col] = {
                value: idx
                for idx, value in enumerate(category_col.cat.categories)
            }

        for col in cat_cols:
            code_dict = cat_code_dict[col]
            code_fillna_value = len(code_dict)
            df[col] = df[col].map(code_dict).fillna(code_fillna_value).astype(
                np.int64)

        # label
        df[label_col] = df[label_col].astype(np.float32)
        return df

    df_groups = {'train': df_train, 'val': df_val, 'test': df_test}

    data_dir = 'onnx_data'
    os.makedirs(data_dir, exist_ok=True)

    for name, df_group in df_groups.items():
        filename = os.path.join(data_dir, f'{name}.csv')
        df_preprocessed = preprocess(
            df_group,
            scaler=StandardScaler,
            num_cols=num_cols,
            cat_cols=cat_cols,
            label_col=label_col)
        df_preprocessed.to_csv(filename, index=False)

    print(df_preprocessed.head())
    print(df_preprocessed.dtypes)

    batch_size = 64
    path_train = os.path.join(data_dir, 'train.csv')
    dataset = TabularDataset(path_train, num_cols, cat_cols, label_col)
    data_loader = DataLoader(dataset, batch_size)

    # our data loader now returns batches of numerical/categorical/label tensor
    num_tensor, cat_tensor, label_tensor = next(iter(data_loader))

    print('numerical value tensor:\n', num_tensor)
    print('categorical value tensor:\n', cat_tensor)
    print('label tensor:\n', label_tensor)

    n_classes = 1
    embedding_size_dict = {
        col: len(code)
        for col, code in cat_code_dict.items()
    }
    embedding_dim_dict = {
        col: embedding_size // 2
        for col, embedding_size in embedding_size_dict.items()
    }
    embedding_dim_dict
    tabular_data_module = TabularDataModule(data_dir, num_cols, cat_cols,
                                            label_col)
    # we can print out the network architecture for inspection
    tabular_model = TabularNet(num_cols, cat_cols, embedding_size_dict,
                               n_classes, embedding_dim_dict)
    print(tabular_model)
    callbacks = [EarlyStopping(monitor='val_loss')]
    trainer = pl.Trainer(max_epochs=8, callbacks=callbacks, gpus=1)
    trainer.fit(tabular_model, tabular_data_module)
    y_true, y_pred = predict(tabular_model, tabular_data_module)
    score = compute_score(y_true, y_pred)
    print(score)
