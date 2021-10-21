import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    root_path = './data/processed_data/house/'
    train_data = pd.read_csv(root_path + 'train_data.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test_data.csv')

    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    target = 'SalePrice'

    nunique = total_data.nunique()
    types = total_data.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in total_data.columns:
        if types[col] == 'object' and nunique[col] < 200:
            print(col, total_data[col].nunique())
            l_enc = LabelEncoder()
            total_data[col] = total_data[col].fillna('VV_likely')
            total_data[col] = l_enc.fit_transform(total_data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
    else:
        total_data.fillna(total_data[col].mean(), inplace=True)

    features = [col for col in total_data.columns if col not in [target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f] for i, f in enumerate(features)
        if f in categorical_columns
    ]

    train_data = total_data.iloc[:len_train]
    test_data = total_data.iloc[len_train:]
    X_train = train_data.drop(target, axis=1).values
    y_train = train_data[target].values.reshape(-1, 1)
    X_test = test_data.drop(target, axis=1).values
    y_test = test_data[target].values.reshape(-1, 1)

    clf = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.2),
        scheduler_params={
            'step_size': 200,  # how to use learning rate scheduler
            'gamma': 0.9
        },
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax'  # "sparsemax"
    )

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_name=['train', 'valid'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=1000,
        patience=200,
        batch_size=1024,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False)

    preds = clf.predict(X_test)
    test_score = mean_squared_error(y_pred=preds, y_true=y_test)
    r2 = r2_score(y_pred=preds, y_true=y_test)
    print(f'FINAL TEST SCORE FOR {r2} : {test_score}')

    regressor = TabNetRegressor(
        scheduler_params={
            'step_size': 200,  # how to use learning rate scheduler
            'gamma': 0.9
        },
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )
    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_name=['train', 'valid'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=1000,
        patience=200,
        batch_size=4096,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False)

    preds = regressor.predict(X_test)
    test_score = mean_squared_error(y_pred=preds, y_true=y_test)
    r2 = r2_score(y_pred=preds, y_true=y_test)
    print(f'FINAL TEST SCORE FOR {r2} : {test_score}')
