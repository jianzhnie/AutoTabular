import numpy as np
import pandas as pd
from autofe.get_feature import *
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    root_path = './data/credit/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'payment'

    classfier = RandomForestClassifier(max_depth=2, random_state=0)

    estimator = LGBMClassifier(objective='binary')

    """RF baseline"""
    # Accuracy: 0.7978011178674529. ROC_AUC: 0.6196475756094981
    total_data_base = get_baseline_total_data(total_data)
    print("random_forest baseline: ")
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
