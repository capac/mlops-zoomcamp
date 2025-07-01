import os
import pandas as pd
from datetime import datetime

import batch

os.environ.get('INPUT_FILE_PATTERN', "s3://nyc-duration/in/{year:04d}'\
               '-{month:02d}.parquet")
os.environ.get('OUTPUT_FILE_PATTERN', "s3://nyc-duration/out/{year:04d}'\
               '-{month:02d}.parquet")

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
    s3_endpoint_url = os.environ.get('S3_ENDPOINT_URL',
                                     'http://localhost:4566')

    options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}

    df = test_dataframe()

    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    print(f'Written to {output_file}')


if __name__ == "__main__":
    year, month = 2023, 1
    output_file = batch.get_output_path(year, month)
    main(year, month)
