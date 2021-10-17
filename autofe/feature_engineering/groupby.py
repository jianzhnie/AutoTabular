import pandas as pd 

def get_category_columns(df, target):
    cat_col_names = []
    for col in df.columns:
        if df[col].dtype in ['object'] and col != target:
            cat_col_names.append(col)
    return cat_col_names


def get_numerical_columns(df, target):
    num_col_names = []
    for col in df.columns:
        if df[col].dtype in ['float', 'int'] and col != target:
            num_col_names.append(col)
    return num_col_names


def get_candidate_numerical_feature(df, target, k):
    num_col_names = get_numerical_columns(df, target)
    num_col_names.append(target)
    k = min(k, len(num_col_names))
    num_data = df[num_col_names]
    num_data = num_data.fillna(0)
    abs_corr = num_data.corr()[target].abs()
    top_k = abs_corr.sort_values(ascending = False)[1:k].index.values.tolist()
    return top_k


def get_candidate_categorical_feature(df, target, threshold):
    cat_col_names = get_category_columns(df, target)
    for col in cat_col_names:
        if df[col].nunique() > len(df)*threshold or df[col].nunique() == 1:
            cat_col_names.remove(col)
    return cat_col_names


def groupby_generate_feature(df, target, threshold, k, methods, reserve = False):
    cat_col_names = get_candidate_categorical_feature(df, target, threshold)
    num_col_names = get_candidate_numerical_feature(df, target, k)

    new_col_names = []
    for cat_col in cat_col_names:
        for num_col in num_col_names:
            for method in methods:
                new_col_name = cat_col + "_" + num_col + "_" + method
                new_col_names.append(new_col_name)
                df[new_col_name] = df.groupby(cat_col)[num_col].transform(method)
                df = df.fillna(0)
    
    if reserve == True:
        return df
    else:
        return df[new_col_names]




