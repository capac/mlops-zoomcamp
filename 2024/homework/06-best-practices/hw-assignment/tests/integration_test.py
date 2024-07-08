import os
import pandas as pd

os.environ.get('INPUT_FILE_PATTERN', "s3://nyc-duration/in/{year:04d}'\
               '-{month:02d}.parquet")
os.environ.get('OUTPUT_FILE_PATTERN', "s3://nyc-duration/out/{year:04d}'\
               '-{month:02d}.parquet")


def get_input_path(year, month):
    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
    default_input_pattern = f'{base_url}/yellow_tripdata_{year:04d}'\
                            f'-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/'\
        'taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    s3_endpoint_url = os.environ.get('S3_ENDPOINT_URL',
                                     'http://localhost:4566')

    options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}

    df = pd.read_parquet('s3://bucket/file.parquet', storage_options=options)
    df.to_parquet(
        input_file,
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


if __name__ == "__main__":
    year, month = 2023, 1
    main(year, month)
