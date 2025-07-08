import os
import pandas as pd
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def generate_input_file(input_file, options):
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


def main(year=2023, month=1):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    os.environ['INPUT_FILE_PATTERN'] = (
        's3://nyc-duration/in/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )
    os.environ['OUTPUT_FILE_PATTERN'] = (
        's3://nyc-duration/out/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    input_file = (
        f's3://nyc-duration/in/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )
    output_file = (
        f's3://nyc-duration/out/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )

    generate_input_file(input_file, options)

    exit_code = os.system(f'python batch.py {year} {month}')
    assert exit_code == 0, f"batch.py exited with code {exit_code}"

    df_result = pd.read_parquet(output_file, storage_options=options)
    total = df_result['predicted_duration'].sum()
    print(f"Sum of predicted durations: {total.round(3)}")


if __name__ == "__main__":
    main()
