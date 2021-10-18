import numpy as np
import pandas as pd
from autofe.get_feature import *
from xgboost import XGBClassifier 

if __name__ == '__main__':
    root_path = './data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'target'

    classfier = XGBClassifier(learning_rate=0.01,
                      n_estimators=100,           # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树
                      colsample_btree=1,         # 所有特征建立决策树
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27,           # 随机数
                      slient = 0
                      )

    estimator = LGBMClassifier(objective='binary')

    """xgboost baseline"""
    # Accuracy: 0.7978011178674529. ROC_AUC: 0.6196475756094981
    total_data_base = get_baseline_total_data(total_data)
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
    """groupby + lr"""
    # Accuracy: 0.8189300411522634. ROC_AUC: 0.8501593726796921
    # threshold = 0.9
    # k = 5
    # methods = ["min", "max", "sum", "mean", "std", "count"]
    # total_data_groupby = get_groupby_total_data(total_data, target_name, threshold, k, methods)
    # total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    # total_data_groupby.to_csv(root_path + 'adult_groupby.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_groupby, target_name, len_train, classfier)

    """GBDT + lr"""
    # AUC: 0.9255204442194576
    # total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    # total_data_GBDT.to_csv(root_path + 'adult_gbdt.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train, classfier)

    """groupby + GBDT + lr"""
    # 加原始特征：AUC: 0.8501569053514051
    # 不加原始特征：AUC: 0.8500834500609618
    # groupby后的特征与原始特征合并，再给GBDT，生成的特征再给lr，AUC: 0.9294256917039849
    # Accuracy: 0.8747619925066028. ROC_AUC: 0.9294256917039849
    # 特征选择后 Accuracy: 0.8755604692586451. ROC_AUC: 0.9304231928022597
    # threshold = 0.9
    # k = 5
    # methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    # groupby_data = get_groupby_total_data(total_data, target_name, threshold,
    #                                       k, methods)
    # total_data_GBDT = get_groupby_GBDT_total_data(groupby_data, target_name)
    # total_data_GBDT = select_feature(total_data_GBDT, target_name, estimator)
    # total_data_GBDT.to_csv(root_path + 'adult_groupby_gbdt.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train,
    #                               classfier)

    """nn embedding + lr"""
    # Accuracy: 0.8492721577298692. ROC_AUC: 0.8992624988473603
    # total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    # total_data_embed.to_csv(root_path + 'adult_embed.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
    #                               classfier)

    """wide & deep embedding + lr"""
    # Accuracy: 0.7777163564891592. ROC_AUC: 0.7640908282089225
    # total_data_embed = get_widedeep_total_data(total_data, target_name)
    # acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
    #                               classfier)

    """AutoFI + lr: simple concate"""
    # threshold = 0.9
    # k = 5
    # methods = ["min", "max", "sum", "mean", "std", "count"]
    # total_data_groupby = get_groupby_total_data(total_data, target_name, threshold, k, methods)
    # total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    # total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    # total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    # total_data = autofi_simple_concat_total_data(total_data_groupby, total_data_GBDT, total_data_embed)
    # total_data.to_csv(root_path + 'adult_autofi.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
    #                               classfier)