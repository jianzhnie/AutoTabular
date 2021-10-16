import gc

from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def get_category_columns(df, target):
    cat_col_names = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category'] and col != target:
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
    if df[target].dtype == 'float':
        select_model = SelectKBest(mutual_info_regression, k=k)
    else:
        select_model = SelectKBest(mutual_info_classif, k=k)
    X = df[num_col_names]
    y = df[target]
    slect_feature_cols = select_model.fit(X, y).get_feature_names_out()
    return slect_feature_cols


def get_candidate_categorical_feature(df, target, threshold):
    cat_col_names = get_category_columns(df, target)
    for col in cat_col_names:
        if df[col].nunique() > len(df) * threshold or df[col].nunique() == 1:
            cat_col_names.remove(col)
    return cat_col_names


def groupby_generate_feature(df, target, threshold, k, methods, reserve=False):
    cat_col_names = get_candidate_categorical_feature(df, target, threshold)
    num_col_names = get_candidate_numerical_feature(df, target, k)

    new_col_names = []
    for cat_col in cat_col_names:
        for num_col in num_col_names:
            for method in methods:
                new_col_name = cat_col + '_' + num_col + '_' + method
                new_col_names.append(new_col_name)
                df[new_col_name] = df.groupby(cat_col)[num_col].transform(
                    method)
                df = df.fillna(0)

    if reserve:
        return df
    else:
        return df[new_col_names]


def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols +
            [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
                columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def add_features_in_group(features, df, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix,
                                       feature_name)] = df[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(
                prefix, feature_name)] = df[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix,
                                       feature_name)] = df[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix,
                                       feature_name)] = df[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix,
                                       feature_name)] = df[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(
                prefix, feature_name)] = df[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix,
                                        feature_name)] = skew(df[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(
                df[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(
                prefix, feature_name)] = df[feature_name].median()
    return features


if __name__ == '__main__':
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X, y = load_iris(return_X_y=True)
    X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    print(df.head())
    print(X_new[:5, ])
    select_cols = get_candidate_numerical_feature(df, target='species', k=3)
    print(df[select_cols].head())
