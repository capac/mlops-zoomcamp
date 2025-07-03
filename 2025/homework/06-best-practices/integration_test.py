import os
from datetime import datetime
import pickle
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_dataframe():
    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    actual_df = pd.DataFrame(data, columns=columns)
    return actual_df


def test_sum_pred(year=2023, month=1):

    df = test_dataframe()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    model_path = os.path.join(os.path.dirname(__file__), 'model.bin')
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    sum_pred_dur = df_result['predicted_duration'].sum().round(2)
    print(f'Sum of predicted durations for the test dataframe: {sum_pred_dur}')

    assert abs(df_result['predicted_duration'].sum() - 92.79) < 1e-1


if __name__ == "__main__":
    year, month = 2023, 1
    test_sum_pred(year, month)
