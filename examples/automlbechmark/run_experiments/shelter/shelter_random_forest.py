
import numpy as np
import pandas as pd
from autofe.get_feature import *
from autofe.deeptabular_utils import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 

if __name__ == '__main__':
    
    root_path = './data/shelter/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'OutcomeType'

    classfier = RandomForestClassifier()
    # estimator = LGBMClassifier(objective='binary')

    """randomforest baseline"""
    # Accuracy: 0.8724279835390947. ROC_AUC: 0.9257845633487581
    # total_data_base = get_baseline_total_data(total_data)
    cat_col_names = get_category_columns(total_data, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data_base = label_encoder.fit_transform(total_data)
    print("randomforest baseline: \n")
    acc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier, task_type='multiclass')



    