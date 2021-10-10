import featuretools as ft
import pandas as pd
from sklearn.datasets import load_iris

# Load data and put into dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})
print(df.head())
print(df.dtypes)

df.ww.init(name='iris')
log_df = df.ww
print(type(log_df))

# Make an entityset and add the entity
es = ft.EntitySet()
print(es)
es.add_dataframe(
    dataframe_name='data', dataframe=df, make_index=True, index='index')
print(es['data'].ww)
# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    max_depth=3,
    target_dataframe_name='data',
    agg_primitives=['mode', 'mean', 'max', 'count'],
    trans_primitives=[
        'add_numeric', 'multiply_numeric', 'cum_min', 'cum_mean', 'cum_max'
    ],
    groupby_trans_primitives=['cum_sum'])

print(feature_defs)
print(feature_matrix.head())
print(feature_matrix.ww)
print('===' * 100)

features = ft.dfs(
    entityset=es, target_dataframe_name='data', features_only=True)
print(features)
feature_matrix = ft.calculate_feature_matrix(features=features, entityset=es)
print(feature_matrix.head())
print('===' * 100)

df = ft.primitives.list_primitives()
trans_primitives = df[df['type'] == 'aggregation']['name'].tolist()
agg_primitives = df[df['type'] == 'transform']['name'].tolist()
print(trans_primitives)
print('===' * 100)
print(agg_primitives)
