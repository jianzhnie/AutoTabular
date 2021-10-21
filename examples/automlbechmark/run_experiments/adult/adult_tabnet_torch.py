import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    root_path = './data/processed_data/adult/'
    test_datafile = root_path + 'test.csv'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')

    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    target = 'target'

    # check that pipeline accepts strings
    total_data.loc[total_data[target] == 0, target] = 'wealthy'
    total_data.loc[total_data[target] == 1, target] = 'not_wealthy'

    nunique = total_data.nunique()
    types = total_data.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in total_data.columns:
        if types[col] == 'object' or nunique[col] < 200:
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
    y_train = train_data[target].values
    X_test = test_data.drop(target, axis=1).values
    y_test = test_data[target].values

    clf = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={
            'step_size': 50,  # how to use learning rate scheduler
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
        eval_metric=['auc'],
        max_epochs=20,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False)
    preds = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)
    preds_valid = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_test)
    print(f'FINAL TEST SCORE FOR {test_auc}')
