import pandas as pd
from autofe.get_feature import (autofi_simple_concat_total_data,
                                get_baseline_total_data, get_GBDT_total_data,
                                get_groupby_GBDT_total_data,
                                get_groupby_total_data,
                                get_nn_embedding_total_data,
                                train_and_evaluate)
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    root_path = './data/processed_data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'target'

    classfier = LogisticRegression(random_state=0)

    estimator = LGBMClassifier(objective='binary')
    """lr baseline"""
    # Accuracy: 0.7978011178674529. ROC_AUC: 0.6196475756094981
    total_data_base = get_baseline_total_data(total_data)
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
    """groupby + lr"""
    # Accuracy: 0.8189300411522634. ROC_AUC: 0.8501593726796921
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_groupby.to_csv(root_path + 'adult_groupby.csv', index=False)
    acc, auc = train_and_evaluate(total_data_groupby, target_name, len_train,
                                  classfier)
    """GBDT + lr"""
    # AUC: 0.9255204442194576
    total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    total_data_GBDT.to_csv(root_path + 'adult_gbdt.csv', index=False)
    acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train,
                                  classfier)
    """groupby + GBDT + lr"""
    # ??????????????????AUC: 0.8501569053514051
    # ?????????????????????AUC: 0.8500834500609618
    # groupby??????????????????????????????????????????GBDT????????????????????????lr???AUC: 0.9294256917039849
    # Accuracy: 0.8747619925066028. ROC_AUC: 0.9294256917039849
    # ??????????????? Accuracy: 0.8755604692586451. ROC_AUC: 0.9304231928022597
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    groupby_data = get_groupby_total_data(total_data, target_name, threshold,
                                          k, methods)
    total_data_GBDT = get_groupby_GBDT_total_data(groupby_data, target_name)
    total_data_GBDT.to_csv(root_path + 'adult_groupby_gbdt.csv', index=False)
    acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train,
                                  classfier)
    """nn embedding + lr"""
    # Accuracy: 0.8492721577298692. ROC_AUC: 0.8992624988473603
    total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    total_data_embed.to_csv(root_path + 'adult_embed.csv', index=False)
    acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
                                  classfier)
    """AutoFI + lr: simple concate"""
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    total_data = autofi_simple_concat_total_data(total_data_groupby,
                                                 total_data_GBDT,
                                                 total_data_embed)
    total_data.to_csv(root_path + 'adult_autofi.csv', index=False)
    acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
                                  classfier)
