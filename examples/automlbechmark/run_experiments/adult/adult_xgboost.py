import pandas as pd
from autofe.get_feature import get_baseline_total_data, get_groupby_total_data, train_and_evaluate
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

if __name__ == '__main__':

    root_path = './data/processed_data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'target'
    classfier = XGBClassifier()
    estimator = LGBMClassifier(objective='binary')
    """xgboost baseline"""
    # Accuracy: 0.8724279835390947. ROC_AUC: 0.9257845633487581
    total_data_base = get_baseline_total_data(total_data)
    print('xgboost baseline: \n')
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)

    param = {
        'subsample': 0.9894965585620157,
        'colsample_bytree': 0.3725398569008492,
        'learning_rate': 0.13098012414701815,
        'max_depth': 4,
        'n_estimators': 500
    }

    classfier = XGBClassifier(**param)
    """xgboost baseline"""
    # Accuracy: 0.8724279835390947. ROC_AUC: 0.9257845633487581
    total_data_base = get_baseline_total_data(total_data)
    print('xgboost baseline: \n')
    acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier)
    """groupby + xgboost"""
    # Accuracy: 0.8727350899821879. ROC_AUC: 0.9254309826594914
    threshold = 0.9
    k = 5
    methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    total_data_groupby = get_groupby_total_data(total_data, target_name,
                                                threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_groupby.to_csv(root_path + 'adult_groupby.csv', index=False)
    print('groupby + xgboost: \n')
    acc, auc = train_and_evaluate(total_data_groupby, target_name, len_train,
                                  classfier)
