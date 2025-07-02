import pandas as pd
from datetime import datetime
import pickle


columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']


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
    return actual_df


def main(year, month):

    df = test_dataframe()
    categorical = ['PULocationID', 'DOLocationID']
    df = df[categorical]
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    sum_pred_dur = df_result['predicted_duration'].sum().round(2)
    print(f'Sum of predicted durations for the test dataframe: {sum_pred_dur}')


if __name__ == "__main__":
    year, month = 2023, 1
    main(year, month)
