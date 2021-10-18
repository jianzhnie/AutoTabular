import pandas as pd
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns

def imputing_missing_features(df, target_name):
    cat_col_names = get_category_columns(df, target_name)
    num_cols_names = get_numerical_columns(df, target_name)
    for col in cat_col_names:
        df[col] = df[col].fillna("None")
    for col in num_cols_names:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def preprocess(df, target_name):
    ### drop duplicates and none rows and cols
    df = df.drop_duplicates()
    print(df)
    row_treshold = int(df.shape[1] * 0.9)
    col_treshold = int(df.shape[0] * 0.9)
    df = df.dropna(axis=0, thresh=row_treshold)
    df = df.dropna(axis=1, thresh=col_treshold)
    df = df.reset_index(drop = True)
    ### imputing missing
    print(df)
    df = imputing_missing_features(df, target_name)
    return df

if __name__ == "__main__":
    root_path = "/home/wenqi-ao/userdata/workdirs/automl_benchmark/data/processed_data/adult/"
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop = True)
    # print(total_data)

    target_name = "target"

    data = preprocess(total_data, target_name)
    print(data)
    
