from pathlib import Path

import pandas as pd
from autofe.get_feature import get_baseline_total_data, get_GBDT_total_data, get_groupby_total_data, train_and_evaluate
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/shelter/'

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    len_train = len(train_data)

    target_name = 'OutcomeType'
    classfier = LogisticRegression(multi_class='multinomial', random_state=0)
    """lr baseline"""
    # Accuracy: 0.5791245791245792.
    total_data_base = get_baseline_total_data(total_data)
    # cat_col_names = get_category_columns(total_data, target_name)
    # label_encoder = LabelEncoder(cat_col_names)
    # total_data_base = label_encoder.fit_transform(total_data)
    print('lr baseline: ')
    acc = train_and_evaluate(
        total_data_base,
        target_name,
        len_train,
        classfier,
        task_type='multiclass')
    """groupby + lr"""
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    acc, auc = train_and_evaluate(
        total_data_groupby,
        target_name,
        len_train,
        classfier,
        task_type='multiclass')
    """GBDT + lr"""
    total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    acc, auc = train_and_evaluate(
        total_data_GBDT,
        target_name,
        len_train,
        classfier,
        task_type='multiclass')
