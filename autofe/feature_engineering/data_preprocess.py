import pandas as pd
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns
from sklearn.preprocessing import StandardScaler


def standardscalar(df, target_name):
    num_col_names = get_numerical_columns(df, target_name)
    scalar = StandardScaler()
    df[num_col_names] = scalar.fit_transform(df[num_col_names])
    return df


def imputing_missing_features(df, target_name):
    cat_col_names = get_category_columns(df, target_name)
    num_cols_names = get_numerical_columns(df, target_name)
    for col in cat_col_names:
        df[col] = df[col].fillna('None')
    for col in num_cols_names:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def drop_data(df, target_name):
    df = df.drop_duplicates()
    row_treshold = int(df.shape[1] * 0.9)
    col_treshold = int(df.shape[0] * 0.6)
    df = df.dropna(axis=0, thresh=row_treshold)
    df = df.dropna(axis=1, thresh=col_treshold)
    df = df.reset_index(drop=True)
    for col in df.columns:
        if 'id' in col.lower() and df[col].nunique() == len(
                df) and col != target_name:
            df = df.drop(col, axis=1)
    return df


def preprocess(df, target_name):
    # drop duplicates and none rows and cols
    df = drop_data(df, target_name)
    # imputing missing
    df = imputing_missing_features(df, target_name)
    # standard scalar
    # df = standardscalar(df, target_name)
    return df


if __name__ == '__main__':
    root_path = './data/house/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    # print(total_data)
    target_name = 'SalePrice'
    data = preprocess(total_data, target_name)
    print(data)
