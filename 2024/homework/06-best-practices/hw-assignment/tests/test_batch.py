import pandas as pd
from datetime import datetime

year, month = 2023, 1
base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
filename = f'{base_url}/yellow_tripdata_{year:04d}-{month:02d}.parquet'
categorical = ['PULocationID', 'DOLocationID']
columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']


def prepare_data(filename, categorical, columns):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df[columns]


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_dataframe():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    actual_df = pd.DataFrame(data, columns=columns)
    expected_df = prepare_data(filename, categorical,
                               columns=columns).iloc[0:len(data)]
    assert actual_df.shape == expected_df.shape
