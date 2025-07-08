import batch
import pandas as pd
from datetime import datetime
from pandas.testing import assert_frame_equal


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    df_columns = ['PULocationID', 'DOLocationID',
                  'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame([
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ], columns=df_columns)

    expected_columns = ['PULocationID', 'DOLocationID',
                        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                        'duration']
    expected = pd.DataFrame([
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0)
    ], columns=expected_columns)

    categorical = ['PULocationID', 'DOLocationID']
    result = batch.prepare_data(df, categorical)

    assert_frame_equal(result, expected)
