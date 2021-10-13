import os
from pathlib import Path

import featuretools as ft
import pandas as pd

SEED = 42

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'

    RESULTS_DIR = ROOTDIR / 'results/adult/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'

    # Make an entityset and add the entity
    es = ft.EntitySet()
    print(es)
    es.add_dataframe(
        dataframe_name='data',
        dataframe=adult_data,
        make_index=True,
        index='index')
    print(es['data'].ww)
    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        max_depth=3,
        target_dataframe_name='data',
        agg_primitives=['mode', 'mean', 'max', 'count'],
        trans_primitives=['cum_min', 'cum_mean', 'cum_max'],
        groupby_trans_primitives=['cum_sum'])

    print(feature_defs)
    print(feature_matrix.head())
    print(feature_matrix.ww)
    print('===' * 100)

    feature_matrix.to_csv(PROCESSED_DATA_DIR / 'adult_ft.csv')

    features = ft.dfs(
        entityset=es, target_dataframe_name='data', features_only=True)
    print(features)
    feature_matrix = ft.calculate_feature_matrix(
        features=features, entityset=es)
    print(feature_matrix.head())
    print('===' * 100)

    df = ft.primitives.list_primitives()
    trans_primitives = df[df['type'] == 'aggregation']['name'].tolist()
    agg_primitives = df[df['type'] == 'transform']['name'].tolist()
    print(trans_primitives)
    print('===' * 100)
    print(agg_primitives)
