import os
from pathlib import Path

import pandas as pd
# from autofe.tabular_embedding.tabular_embedding_transformer import TabularEmbeddingTransformer
from autogluon.tabular import TabularPredictor
from pytorch_widedeep.utils import LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('/home/wenqi-ao/userdata/workdirs/automl_benchmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'
    init_args = {'eval_metric': 'roc_auc', 'path': RESULTS_DIR}

    cat_col_names = []
    for col in adult_data.columns:
        if adult_data[col].dtype == 'object':
            cat_col_names.append(col)

    num_col_names = []
    for col in adult_data.columns:
        if adult_data[col].dtype in ['float', 'int'] and col != 'target':
            num_col_names.append(col)
    
    #calculate corr to get candidate numerical feature columns
    num_data = adult_data[num_col_names + [target_name]]
    num_data = num_data.fillna(0)
    k = min(5, len(num_col_names))
    abs_corr = num_data.corr()[target_name].abs()
    top_k = abs_corr.sort_values(ascending = False)[1:k+1].index.values.tolist()
    cand_num_col = top_k

    #get candidate categorical feature columns
    for col in cat_col_names:
        if adult_data[col].nunique() > len(adult_data)*0.95 or adult_data[col].nunique() == 1:
            cat_col_names.remove(col)
    
    for cat_col in cat_col_names:
        for num_col in num_col_names:
            for method in ["min", "max", "sum", "mean", "std", "count"]:
                new_col_name = cat_col + "_" + num_col + "_" + method
                adult_data[new_col_name] = adult_data.groupby(cat_col)[num_col].transform(method)
    print(adult_data.head(5))


    num_classes = adult_data[target_name].nunique()

    print(num_classes)
    print(cat_col_names)
    print(num_col_names)
    print(adult_data.info())
    print(adult_data.describe())

    label_encoder = LabelEncoder(cat_col_names)
    adult_data = label_encoder.fit_transform(adult_data)

    IndList = range(X.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = adult_data.iloc[train_list]
    val = adult_data.iloc[val_list]
    test = adult_data.iloc[test_list]

    predictor = TabularPredictor(
        label=target_name, path=RESULTS_DIR).fit(
            train_data=train, tuning_data=val)

    scores = predictor.evaluate(test, auxiliary_metrics=False)
    leaderboard = predictor.leaderboard(test)

