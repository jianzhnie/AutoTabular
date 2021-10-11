import os
from pathlib import Path

import pandas as pd
from autofe.tabular_embedding.tabular_embedding_transformer import TabularEmbeddingTransformer
from sklearn.datasets import fetch_covtype

if __name__ == '__main__':
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    datafile = ROOTDIR / 'data/covtype.data.gz'
    RESULTS_DIR = ROOTDIR / 'results/covtype/autogluon'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    data_X, data_y = fetch_covtype(as_frame=True, return_X_y=True)
    target_name = ['Covertype']
    cat_col_names = [
        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
    ]

    num_col_names = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]

    date_col_names = []
    feature_columns = (
        num_col_names + cat_col_names + date_col_names + target_name)

    print(data_X)
    print(data_X.info())
    print(data_X.describe())
    data_y = pd.DataFrame(data_y)

    # GBDT embeddings
    transformer = TabularEmbeddingTransformer(
        cat_col_names=cat_col_names,
        num_col_names=num_col_names,
        date_col_names=[],
        target_name=target_name,
        num_classes=2)
