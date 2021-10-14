import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from autofe.feature_engineering.groupby import groupby_generate_feature, get_category_columns
from autofe.feature_engineering.gbdt_feature import LightGBMFeatureTransformer
from autofe.deeptabular_utils import LabelEncoder


root_path = "/home/wenqi-ao/userdata/workdirs/automl_benchmark/data/processed_data/adult/"
train_data = pd.read_csv(root_path + 'train.csv')
len_train = len(train_data)
test_data = pd.read_csv(root_path + 'test.csv')
total_data = pd.concat([train_data, test_data]).reset_index(drop = True)

target_name = "target"

"""lr baseline"""
# total_data = pd.get_dummies(total_data).fillna(0)

"""groupby + lr"""
##### AUC: 0.850158787211963
# threshold = 0.9
# k = 5
# methods = ["min", "max", "sum", "mean", "std", "count"]
# generated_feature = groupby_generate_feature(total_data, target_name, threshold, k, methods)
# total_data = pd.concat([total_data, generated_feature], axis = 1)

"""GBDT + lr"""
##### AUC: 0.9255204442194576
# cat_col_names = get_category_columns(total_data, target_name)
# label_encoder = LabelEncoder(cat_col_names)
# total_data = label_encoder.fit_transform(total_data)
# clf = LightGBMFeatureTransformer(
#         task='classification', categorical_feature=cat_col_names, params={
#                      'n_estimators': 100,
#                      'max_depth': 3
#                  })
# X = total_data.drop(target_name, axis=1)
# y = total_data[target_name]
# clf.fit(X, y)
# X_enc = clf.concate_transform(X, concate=False)
# total_data = pd.concat([X_enc, y], axis = 1)
# total_data = pd.concat([total_data, X_enc], axis = 1).fillna(0)

"""groupby + GBDT + lr"""
##### 加原始特征：AUC: 0.8501569053514051
##### 不加原始特征：AUC: 0.8500834500609618
threshold = 0.9
k = 5
methods = ["min", "max", "sum", "mean", "std", "count"]
generated_feature = groupby_generate_feature(total_data, target_name, threshold, k, methods)
total_data = pd.concat([total_data, generated_feature], axis = 1)
print(total_data.head(5))

cat_col_names = get_category_columns(total_data, target_name)
label_encoder = LabelEncoder(cat_col_names)
total_data = label_encoder.fit_transform(total_data)
clf = LightGBMFeatureTransformer(
        task='classification', categorical_feature=cat_col_names, params={
                     'n_estimators': 100,
                     'max_depth': 3
                 })
X = total_data.drop(target_name, axis=1)
y = total_data[target_name]
clf.fit(X, y)
X_enc = clf.concate_transform(X, concate=False)
total_data = pd.concat([generated_feature, X_enc, y], axis = 1)


#train and evaluate
train_data = total_data.iloc[:len_train]
test_data = total_data.iloc[len_train:]
X_train = train_data.drop(target_name, axis=1)
y_train = train_data[target_name]
X_test = test_data.drop(target_name, axis=1)
y_test = test_data[target_name]
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
pred = lr.predict_proba(X_test)[:, 1]
print(f"Accuracy: {lr.score(X_test, y_test)}")
print(f"AUC: {roc_auc_score(y_test, pred)}")