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
print(df)
# Make an entityset and add the entity
es = ft.EntitySet(id='iris')
print(es)
es.entity_from_dataframe(
    entity_id='data', dataframe=df, make_index=True, index='index')

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='data',
    trans_primitives=['add_numeric', 'multiply_numeric'],
    agg_primitives=None)

print(feature_defs)
print(feature_matrix.head())
print('===' * 100)

df = ft.primitives.list_primitives()
trans_primitives = df[df['type'] == 'aggregation']['name'].tolist()
agg_primitives = df[df['type'] == 'transform']['name'].tolist()
print(trans_primitives)
print('===' * 100)
print(agg_primitives)
# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='data',
    trans_primitives=trans_primitives,
    agg_primitives=agg_primitives)

print(feature_defs)
print(feature_matrix.head())
