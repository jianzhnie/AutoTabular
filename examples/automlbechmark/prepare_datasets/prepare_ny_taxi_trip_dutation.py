import os
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from geopy.distance import geodesic

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / 'raw_data/nyc_taxi/'
PROCESSED_DATA_DIR = ROOT_DIR / 'processed_data/nyc_taxi/'

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

nyc_taxi = pd.read_csv(
    RAW_DATA_DIR / 'train_extended.csv',
    parse_dates=['pickup_datetime', 'dropoff_datetime'],
)
nyc_taxi = nyc_taxi[nyc_taxi.passenger_count.between(1,
                                                     6)].reset_index(drop=True)
nyc_taxi.drop('id', axis=1, inplace=True)

# Chronological split
nyc_taxi = nyc_taxi.sort_values('pickup_datetime').reset_index(drop=True)
test_size = int(np.ceil(nyc_taxi.shape[0] * 0.1))
train_size = nyc_taxi.shape[0] - test_size * 2

# train
nyc_taxi_train = nyc_taxi.iloc[:train_size].reset_index(drop=True)
tmp = nyc_taxi.iloc[train_size:].reset_index(drop=True)

# valid and test
nyc_taxi_val = tmp.iloc[:test_size].reset_index(drop=True)
nyc_taxi_test = tmp.iloc[test_size:].reset_index(drop=True)

nyc_taxi_train['dset'] = 0
nyc_taxi_val['dset'] = 1
nyc_taxi_test['dset'] = 2

nyc_taxi = pd.concat([nyc_taxi_train, nyc_taxi_val, nyc_taxi_test])

del (nyc_taxi_train, nyc_taxi_val, nyc_taxi_test)

remove_index_cols = ['day_period', 'month', 'season', 'day_name']
for col in remove_index_cols:
    nyc_taxi[col] = nyc_taxi[col].apply(lambda x: x.split('.')[-1])

txt_cols = [
    'pickup_neighbourhood',
    'dropoff_district',
    'dropoff_neighbourhood',
    'day_period',
    'month',
    'season',
    'weekday_or_weekend',
    'regular_day_or_holiday',
    'day_name',
]
for col in txt_cols:
    nyc_taxi[col] = nyc_taxi[col].str.lower()

neighbourhood_cols = ['pickup_neighbourhood', 'dropoff_neighbourhood']
for col in neighbourhood_cols:
    nyc_taxi[col] = nyc_taxi[col].apply(
        lambda x: x.replace(' ', '_').replace('-', '_'))

nyc_taxi['day_of_month'] = nyc_taxi.pickup_datetime.dt.day


def distance_travelled(coords):
    return geodesic((coords[0], coords[1]), (coords[2], coords[3])).km


start_lats = nyc_taxi.pickup_latitude.tolist()
start_lons = nyc_taxi.pickup_longitude.tolist()
end_lats = nyc_taxi.dropoff_latitude.tolist()
end_lons = nyc_taxi.dropoff_longitude.tolist()

s = time()
with Pool(8) as p:
    distances = p.map(distance_travelled,
                      zip(start_lats, start_lons, end_lats, end_lons))
e = time() - s
print('computing distances took {} secs'.format(e))

nyc_taxi['distance_travelled'] = distances

nyc_taxi['pickup_x'] = np.cos(nyc_taxi.pickup_latitude) * np.cos(
    nyc_taxi.pickup_longitude)
nyc_taxi['dropoff_x'] = np.cos(nyc_taxi.dropoff_latitude) * np.cos(
    nyc_taxi.dropoff_longitude)
nyc_taxi['pickup_y'] = np.cos(nyc_taxi.pickup_longitude) * np.sin(
    nyc_taxi.pickup_longitude)
nyc_taxi['dropoff_y'] = np.cos(nyc_taxi.dropoff_longitude) * np.sin(
    nyc_taxi.dropoff_longitude)
nyc_taxi['pickup_z'] = np.sin(nyc_taxi.pickup_latitude)
nyc_taxi['dropoff_z'] = np.sin(nyc_taxi.dropoff_latitude)
nyc_taxi['pickup_latitude'] = nyc_taxi.pickup_latitude / 60
nyc_taxi['dropoff_latitude'] = nyc_taxi.dropoff_latitude / 60
nyc_taxi['pickup_longitude'] = nyc_taxi.pickup_longitude / 180
nyc_taxi['dropoff_longitude'] = nyc_taxi.dropoff_longitude / 180

# I know we have train_duration in the data, but just for sanity
nyc_taxi['target'] = (nyc_taxi.dropoff_datetime -
                      nyc_taxi.pickup_datetime).astype('timedelta64[s]')

nyc_taxi_train = nyc_taxi[nyc_taxi.dset == 0].drop('dset', axis=1)
nyc_taxi_val = nyc_taxi[nyc_taxi.dset == 1].drop('dset', axis=1)
nyc_taxi_test = nyc_taxi[nyc_taxi.dset == 2].drop('dset', axis=1)

nyc_taxi_train.to_pickle(PROCESSED_DATA_DIR / 'nyc_taxi_train.p')
nyc_taxi_val.to_pickle(PROCESSED_DATA_DIR / 'nyc_taxi_val.p')
nyc_taxi_test.to_pickle(PROCESSED_DATA_DIR / 'nyc_taxi_test.p')
