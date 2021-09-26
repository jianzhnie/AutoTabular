from sklearn.datasets import load_iris
import pandas as pd
import featuretools as ft

# Load data and put into dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Make an entityset and add the entity
es = ft.EntitySet(id='iris')
es.entity_from_dataframe(
    entity_id='data', dataframe=df, make_index=True, index='index')

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='data',
    trans_primitives=['add_numeric', 'multiply_numeric'])

feature_matrix.head()