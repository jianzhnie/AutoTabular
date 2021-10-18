import os
import pandas as pd

from autofe.feature_engineering.data_preprocess import preprocess, split_train_test


if __name__ == "__main__":
    root_dir = "./data/house/"
    data = pd.read_csv(root_dir + 'train.csv')
    target_name = "SalePrice"
    data = preprocess(data, target_name)
    data_train, data_test = split_train_test(data, target_name, 0.2)
    data_train.to_csv(root_dir + 'data_train.csv', index = False)
    data_test.to_csv(root_dir + 'data_test.csv', index = False)
