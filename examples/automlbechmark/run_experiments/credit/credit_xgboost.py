
import numpy as np
import pandas as pd
from autofe.get_feature import *
from xgboost import XGBClassifier 

if __name__ == '__main__':
    
    root_path = './data/credit/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'payment'

    classfier = XGBClassifier()
    estimator = LGBMClassifier(objective='binary')

    """xgboost baseline"""
    # Accuracy: 0.8724279835390947. ROC_AUC: 0.9257845633487581
    total_data_base = get_baseline_total_data(total_data)
    print("xgboost baseline: \n")
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)



    