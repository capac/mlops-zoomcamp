import os
import pandas as pd
from datetime import datetime

# import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)


# input_file = batch.get_input_path(2023, 1)
output_file = 's3://nyc-duration/out/2023-01.parquet'
# output_file = batch.get_output_path(2023, 1)

# df_input.to_parquet(
#     input_parquet_file,
#     engine='pyarrow',
#     compression=None,
#     index=False,
#     storage_options=options
# )


# os.system('python batch.py 2023 1')


df_actual = pd.read_parquet(output_file, storage_options=options)

y_pred = df_actual['predicted_duration']
pred_dur_sum = y_pred.sum().round(2)
print(f'Sum of predicted durations: {pred_dur_sum}')

# assert abs(df_actual['predicted_duration'].sum() - 92.3) < 0.1
