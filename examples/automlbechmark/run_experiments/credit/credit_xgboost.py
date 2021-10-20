from pathlib import Path

import pandas as pd
from autofe.get_feature import get_baseline_total_data, train_and_evaluate
from xgboost import XGBClassifier

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/credit/'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    len_train = len(train_data)
    target_name = 'payment'

    classfier = XGBClassifier()
    """xgboost baseline"""
    # Accuracy: 0.8724279835390947. ROC_AUC: 0.9257845633487581
    total_data_base = get_baseline_total_data(total_data)
    print('xgboost baseline: \n')
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
