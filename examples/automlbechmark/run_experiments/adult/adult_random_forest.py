import pandas as pd
from autofe.deeptabular_utils import LabelEncoder
from autofe.feature_engineering.groupby import get_category_columns
from autofe.get_feature import (generate_cross_feature,
                                get_baseline_total_data, get_cross_columns,
                                get_GBDT_total_data, get_groupby_total_data,
                                train_and_evaluate)
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    root_path = './data/processed_data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    target_name = 'target'

    classfier = RandomForestClassifier(random_state=0)
    """RF baseline"""
    total_data_base = get_baseline_total_data(total_data)
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
    """RF baseline_labelencoder"""
    cat_col_names = get_category_columns(total_data, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data_labelencoder = label_encoder.fit_transform(total_data)
    acc, auc = train_and_evaluate(total_data_labelencoder, target_name,
                                  len_train, classfier)
    # cross data
    cat_col_names = get_category_columns(total_data, target_name)
    crossed_cols = get_cross_columns(cat_col_names)
    total_cross_data = generate_cross_feature(
        total_data, crossed_cols=crossed_cols)
    print(total_cross_data)
    cat_col_names = get_category_columns(total_cross_data, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_cross_data = label_encoder.fit_transform(total_cross_data)
    acc, auc = train_and_evaluate(total_cross_data, target_name, len_train,
                                  classfier)
    """groupby + RF"""
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_groupby.to_csv(root_path + 'adult_groupby.csv', index=False)
    acc, auc = train_and_evaluate(total_data_groupby, target_name, len_train,
                                  classfier)
    """GBDT + RF"""
    total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    total_data_GBDT.to_csv(root_path + 'adult_gbdt.csv', index=False)
    acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train,
                                  classfier)
