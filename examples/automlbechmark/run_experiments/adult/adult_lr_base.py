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
from sklearn.metrics import accuracy_score, roc_auc_score
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
        input_dim=36)

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
    total_data = pd.concat([X_vec, df[target_name]],
                           axis=1,
                           verify_integrity=True)
    return total_data


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
    total_data = pd.concat([df_groupby, df_gbtd, df_embedding], axis = 1)
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


def train_and_evaluate(total_data, target_name, num_train_set, classfier):
    train_data = total_data.iloc[:num_train_set]
    test_data = total_data.iloc[num_train_set:]
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    clf = classfier.fit(X_train, y_train)
    preds = preds = clf.predict(X_test)
    preds_prob = classfier.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    print(f'Accuracy: {acc}. ROC_AUC: {auc}')
    return acc, auc


if __name__ == '__main__':
    root_path = '/home/wenqi-ao/userdata/workdirs/automl_benchmark/data/processed_data/adult/'
    saved_dir = '/home/wenqi-ao/userdata/workdirs/automl_benchmark/data/processed_data/adult/'
    train_data = pd.read_csv(root_path + 'train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'target'

    classfier = LogisticRegression(random_state=0)
    estimator = LGBMClassifier(objective='binary')

    """lr baseline"""
    # Accuracy: 0.7978011178674529. ROC_AUC: 0.6196475756094981
    # total_data_base = get_baseline_total_data(total_data)
    # acc, auc = train_and_evaluate(total_data_base, target_name, len_train,
    #                               classfier)
    """groupby + lr"""
    # AUC: 0.850158787211963
    # threshold = 0.9
    # k = 5
    # methods = ["min", "max", "sum", "mean", "std", "count"]
    # total_data_groupby = get_groupby_total_data(total_data, target_name, threshold, k, methods)
    # total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    # total_data_groupby.to_csv(saved_dir + 'adult_groupby.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_groupby, target_name, len_train, classfier)

    """GBDT + lr"""
    # AUC: 0.9255204442194576
    # total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    # total_data_GBDT.to_csv(saved_dir + 'adult_gbdt.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train, classfier)

    """groupby + GBDT + lr"""
    # 加原始特征：AUC: 0.8501569053514051
    # 不加原始特征：AUC: 0.8500834500609618
    # groupby后的特征与原始特征合并，再给GBDT，生成的特征再给lr，AUC: 0.9294256917039849
    # Accuracy: 0.8747619925066028. ROC_AUC: 0.9294256917039849
    # 特征选择后 Accuracy: 0.8755604692586451. ROC_AUC: 0.9304231928022597
    # threshold = 0.9
    # k = 5
    # methods = ['min', 'max', 'sum', 'mean', 'std', 'count']
    # groupby_data = get_groupby_total_data(total_data, target_name, threshold,
    #                                       k, methods)
    # total_data_GBDT = get_groupby_GBDT_total_data(groupby_data, target_name)
    # total_data_GBDT = select_feature(total_data_GBDT, target_name, estimator)
    # total_data_GBDT.to_csv(saved_dir + 'adult_groupby_gbdt.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_GBDT, target_name, len_train,
    #                               classfier)

    """nn embedding + lr"""
    # Accuracy: 0.8492721577298692. ROC_AUC: 0.8992624988473603
    # total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    # total_data_embed.to_csv(saved_dir + 'adult_embed.csv', index = False)
    # acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
    #                               classfier)

    """wide & deep embedding + lr"""
    # Accuracy: 0.7777163564891592. ROC_AUC: 0.7640908282089225
    # total_data_embed = get_widedeep_total_data(total_data, target_name)
    # acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
    #                               classfier)

    """AutoFI + lr: simple concate"""
    threshold = 0.9
    k = 5
    methods = ["min", "max", "sum", "mean", "std", "count"]
    total_data_groupby = get_groupby_total_data(total_data, target_name, threshold, k, methods)
    total_data_groupby = pd.get_dummies(total_data_groupby).fillna(0)
    total_data_GBDT = get_GBDT_total_data(total_data, target_name)
    total_data_embed = get_nn_embedding_total_data(total_data, target_name)
    total_data = autofi_simple_concat_total_data(total_data_groupby, total_data_GBDT, total_data_embed)
    total_data.to_csv(saved_dir + 'adult_autofi.csv', index = False)
    acc, auc = train_and_evaluate(total_data_embed, target_name, len_train,
                                  classfier)