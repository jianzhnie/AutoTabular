import os
import pandas as pd

from autofe.feature_engineering.data_preprocess import preprocess

if __name__ == "__main__":
    root_dir = "./data/house/"
    train_data = pd.read_csv(root_dir + 'train.csv')
    test_data = pd.read_csv(root_dir + 'test.csv')

    target_name = "SalePrice"
    train_data = preprocess(train_data, target_name)
    test_data = preprocess(test_data, target_name)

    train_data.to_csv(root_dir + 'train.csv', index = False)
    test_data.to_csv(root_dir + 'test.csv', index = False)