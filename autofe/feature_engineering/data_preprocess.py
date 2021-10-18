import pandas as pd
from sklearn.model_selection import train_test_split
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns

def split_train_test(df, target_name, test_size):
    X = df.drop(target_name, axis=1)
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state=2021)
    data_train = pd.concat([X_train, y_train], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)
    return data_train, data_test


def imputing_missing_features(df, target_name):
    cat_col_names = get_category_columns(df, target_name)
    num_cols_names = get_numerical_columns(df, target_name)
    for col in cat_col_names:
        df[col] = df[col].fillna("None")
    for col in num_cols_names:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def drop_data(df, target_name):
    df = df.drop_duplicates()
    row_treshold = int(df.shape[1] * 0.9)
    col_treshold = int(df.shape[0] * 0.6)
    df = df.dropna(axis=0, thresh=row_treshold)
    df = df.dropna(axis=1, thresh=col_treshold)
    df = df.reset_index(drop = True)
    for col in df.columns:
        if 'id' in col.lower() and df[col].nunique() == len(df) and col != target_name:
            df = df.drop(col, axis=1)
    return df


def preprocess(df, target_name):
    ### drop duplicates and none rows and cols
    df = drop_data(df, target_name)
    ### imputing missing
    df = imputing_missing_features(df, target_name)
    return df

if __name__ == "__main__":
    root_path = "./data/house/"
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop = True)
    # print(total_data)

    target_name = "SalePrice"

    data = preprocess(total_data, target_name)
    print(data)
    
