import flash
import numpy as np
import pandas as pd
import torch
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns
from autofe.get_feature import generate_cross_feature, get_cross_columns, get_groupby_total_data
from flash.tabular import TabularClassificationData, TabularClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

if __name__ == '__main__':
    root_path = './data/processed_data/adult/'
    test_datafile = root_path + 'test.csv'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    target_name = 'target'

    cat_col_names = get_category_columns(total_data, target_name)
    num_col_names = get_numerical_columns(total_data, target_name)

    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]

    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    # tabnet
    # 1. Create the DataModule
    datamodule = TabularClassificationData.from_data_frame(
        categorical_fields=cat_col_names,
        numerical_fields=num_col_names,
        target_fields=target_name,
        train_data_frame=train_data,
        val_data_frame=test_data,
        batch_size=128,
    )
    # 2. Build the task
    model = TabularClassifier.from_data(datamodule)
    # 3. Create the trainer and train the model
    trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
    trainer.fit(model, datamodule=datamodule)
    # 4. Generate predictions from a CSV
    preds_mat = model.predict(test_datafile)
    preds_mat = np.array(preds_mat)
    preds_prob = preds_mat[:, 1]
    print(preds_mat.shape)
    preds = np.argmax(preds_mat, axis=1)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(type(preds))
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')

    # tabnet + groupby
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    cross_col_names = get_cross_columns(cat_col_names)
    total_data = generate_cross_feature(
        total_data, crossed_cols=cross_col_names)

    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_groupby.to_csv(root_path + 'adult_groupby.csv', index=False)

    cat_col_names = get_category_columns(total_data_groupby, target_name)
    num_col_names = get_numerical_columns(total_data_groupby, target_name)

    train_data = total_data_groupby.iloc[:len_train]
    test_data = total_data_groupby.iloc[len_train:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    test_data.to_csv(test_datafile, index=None)

    # tabnet
    # 1. Create the DataModule
    datamodule = TabularClassificationData.from_data_frame(
        categorical_fields=cat_col_names,
        numerical_fields=num_col_names,
        target_fields=target_name,
        train_data_frame=train_data,
        val_data_frame=test_data,
        batch_size=128,
    )
    # 2. Build the task
    model = TabularClassifier.from_data(datamodule)
    # 3. Create the trainer and train the model
    trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
    trainer.fit(model, datamodule=datamodule)
    # 4. Generate predictions from a CSV
    preds_mat = model.predict(test_datafile)
    preds_mat = np.array(preds_mat)
    preds_prob = preds_mat[:, 1]
    print(preds_mat.shape)
    preds = np.argmax(preds_mat, axis=1)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    print(type(preds))
    print(f'Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}')
