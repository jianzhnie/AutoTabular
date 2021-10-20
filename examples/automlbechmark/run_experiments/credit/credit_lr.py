from pathlib import Path

import pandas as pd
from autofe.get_feature import get_baseline_total_data, train_and_evaluate
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/credit/'

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    len_train = len(train_data)

    target_name = 'payment'
    classfier = LogisticRegression(random_state=0)
    """lr baseline"""
    # Accuracy: 0.7821666666666667. ROC_AUC: 0.6457363528451024
    total_data_base = get_baseline_total_data(total_data)
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
