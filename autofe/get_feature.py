import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from autofe.deeptabular_utils import LabelEncoder
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from autofe.feature_engineering.groupby import get_category_columns, get_numerical_columns, groupby_generate_feature

from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.feature_selection import SelectFromModel


def get_baseline_total_data(df):
    return pd.get_dummies(df).fillna(0)


def get_groupby_total_data(
        df,
        target_name,
        threshold=0.9,
        k=5,
        methods=['min', 'max', 'sum', 'mean', 'std', 'count'],
        reserve=True):
    generated_feature = groupby_generate_feature(df, target_name, threshold, k,
                                                 methods, reserve)
    return generated_feature


def generate_cross_feature(df: pd.DataFrame, crossed_cols, keep_all=True):
    df_cc = df.copy()
    crossed_colnames = []
    for cols in crossed_cols:
        for c in cols:
            df_cc[c] = df_cc[c].astype('str')
        colname = '_'.join(cols)
        df_cc[colname] = df_cc[list(cols)].apply(lambda x: '_'.join(x), axis=1)

        crossed_colnames.append(colname)
    if keep_all:
        return df_cc
    else:
        return df_cc[crossed_colnames]


def get_cross_columns(category_cols):
    crossed_cols = []
    for i in range(0, len(category_cols) - 1):
        for j in range(i + 1, len(category_cols)):
            crossed_cols.append((category_cols[i], category_cols[j]))
    return crossed_cols


def get_GBDT_total_data(df, target_name):
    cat_col_names = get_category_columns(df, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data = label_encoder.fit_transform(df)
    clf = LightGBMFeatureTransformer(
        task='classification',
        categorical_feature=cat_col_names,
        params={
            'n_estimators': 100,
            'max_depth': 3
        })
    X = total_data.drop(target_name, axis=1)
    y = total_data[target_name]
    clf.fit(X, y)
    X_enc = clf.concate_transform(X, concate=False)
    total_data = pd.concat([X_enc, y], axis=1)
    return total_data


def get_groupby_GBDT_total_data(groupby_df, target_name):
    cat_col_names = get_category_columns(groupby_df, target_name)
    label_encoder = LabelEncoder(cat_col_names)
    total_data = label_encoder.fit_transform(groupby_df)
    clf = LightGBMFeatureTransformer(
        task='classification',
        categorical_feature=cat_col_names,
        params={
            'n_estimators': 100,
            'max_depth': 3
        })
    X = total_data.drop(target_name, axis=1)
    y = total_data[target_name]
    clf.fit(X, y)
    X_enc = clf.concate_transform(X, concate=False)
    total_data = pd.concat([X_enc, y], axis=1)
    return total_data


def get_nn_embedding_total_data(df, target_name):
    target = df[target_name].values
    cat_col_names = get_category_columns(df, target_name)
    num_cols_names = get_numerical_columns(df, target_name)

    wide_cols = cat_col_names
    crossed_cols = []
    for i in range(0, len(wide_cols) - 1):
        for j in range(i + 1, len(wide_cols)):
            crossed_cols.append((wide_cols[i], wide_cols[j]))
    wide_prprocessor = WidePreprocessor(wide_cols, crossed_cols)
    X_wide = wide_prprocessor.fit_transform(df)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=num_cols_names,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(df)

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=32)

    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)
    model = WideDeep(wide=wide, deeptabular=ft_transformer)
    trainer = Trainer(model, objective='binary', metrics=[Accuracy])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=30,
        batch_size=512,
        val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec = t2v.transform(df)
    feature_names = ['nn_embed_' + str(i) for i in range(X_vec.shape[1])]
    X_vec = pd.DataFrame(X_vec, columns=feature_names)
    return X_vec


def get_widedeep_total_data(df, target_name):
    cat_col_names = get_category_columns(df, target_name)
    num_cols_names = get_numerical_columns(df, target_name)

    wide_cols = cat_col_names
    crossed_cols = []
    for i in range(0, len(wide_cols) - 1):
        for j in range(i + 1, len(wide_cols)):
            crossed_cols.append((wide_cols[i], wide_cols[j]))
    wide_prprocessor = WidePreprocessor(wide_cols, crossed_cols)
    X_wide = wide_prprocessor.fit_transform(df)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=num_cols_names,
        for_transformer=True)
    X_tab = tab_preprocessor.fit_transform(df)

    feature_names = ['wide_embed_' + str(i) for i in range(X_wide.shape[1])]
    X_wide = pd.DataFrame(X_wide, columns=feature_names)

    feature_names = ['tab_embed_' + str(i) for i in range(X_tab.shape[1])]
    X_tab = pd.DataFrame(X_tab, columns=feature_names)

    total_data = pd.concat([X_wide, X_tab, df[target_name]],
                           axis=1,
                           verify_integrity=True)
    return total_data


def autofi_simple_concat_total_data(df_groupby, df_gbtd, df_embedding):
    total_data = pd.concat([df_groupby, df_gbtd, df_embedding], axis=1)
    total_data = total_data.loc[:, ~total_data.columns.duplicated()]
    return total_data


def select_feature(df, target_name, estimator):
    X = df.drop(target_name, axis=1)
    y = df[target_name]
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    support = selector.get_support()
    col_names = X.columns[support]
    X = selector.transform(X)
    X = pd.DataFrame(X, columns=col_names)
    total_data = X
    total_data[target_name] = y
    return total_data


def train_and_evaluate(total_data,
                       target_name,
                       num_train_set,
                       classifier,
                       task_type='binary'):
    train_data = total_data.iloc[:num_train_set]
    test_data = total_data.iloc[num_train_set:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    clf = classifier.fit(X_train, y_train)
    preds = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        preds_prob = classifier.predict_proba(X_test)[:, 1]
    if task_type == 'binary':
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds_prob)
        print(f'Accuracy: {acc}. ROC_AUC: {auc}')
        return acc, auc
    elif task_type == 'multiclass':
        acc = accuracy_score(y_test, preds)
        print(f'Accuracy: {acc}.')
        return acc
    else:
        score = r2_score(y_test, preds)
        print(f'r2_score: {score}')
        return score