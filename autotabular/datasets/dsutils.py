import os

basedir = os.path.dirname(__file__)


def load_boston():
    from sklearn import datasets
    import pandas as pd
    boston_dataset = datasets.load_boston()
    data = pd.DataFrame(boston_dataset.data)
    data.columns = boston_dataset.feature_names
    data.insert(0, 'target', boston_dataset.target)
    return data


def load_heart_disease_uci():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/heart-disease-uci.csv')
    return data


def load_titanic():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/Titanic.csv')
    return data


def load_bank():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/bank-uci.csv.gz')
    return data


def load_bank_by_dask():
    from dask import dataframe as dd
    data = dd.read_csv(
        f'{basedir}/data/bank-uci.csv.gz', compression='gzip', blocksize=None)
    return data


def load_adult():
    import pandas as pd
    # print(f'Base dir:{basedir}')
    data = pd.read_csv(
        f'{basedir}/data/adult-uci.csv.gz', compression='gzip', header=None)
    return data


def load_glass_uci():
    import pandas as pd
    # print(f'Base dir:{basedir}')
    data = pd.read_csv(f'{basedir}/data/glass_uci.csv', header=None)
    return data


def load_blood():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/blood.csv')
    return data


def load_telescope():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/telescope.csv')
    return data


def load_Bike_Sharing():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/Bike_Sharing.csv')
    return data


def load_movielens():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/data/movielens_sample.txt')
    return data
