#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd


def get_input_path(year, month):
    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/'
    default_input_pattern = (
        f'{base_url}yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )
    # To use the Localstack S3 bucket, use the following input file
    # pattern export command: export INPUT_FILE_PATTERN=\
    # "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    filename = input_pattern.format(year=year, month=month)
    print(f'Input filename: {filename}')
    return filename


def get_output_path(year, month):
    default_output_pattern = (
        f's3://nyc-duration/out/yellow_tripdata_{year:04d}-{month:02d}.parquet'
        )
    # To download locally use the following output file pattern export command:
    # export OUTPUT_FILE_PATTERN=\
    # "output/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    filename = output_pattern.format(year=year, month=month)
    print(f'Output filename: {filename}')
    return filename


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def read_data(filename, categorical):
    # To download from NYC Taxi Data website, leave S3_ENDPOINT_URL empty.
    # In Fish that requires 'set -e S3_ENDPOINT_URL', else to set the variable
    # use 'export S3_ENDPOINT_URL="http://localhost:4566"'. This determines
    # whether 'read_data' downloads the file from the website or simply reads
    # it from the Localstack S3 bucket (remember to launch Docker and run
    # 'docker-compose up', or 'docker-compose up --build' if it's the first
    # time you run Docker or you need to rebuild the container.
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)


def save_data(filename, df):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    df.to_parquet(filename, engine='pyarrow',
                  index=False, storage_options=options)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('Predicted mean duration:', y_pred.mean().round(3))

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(output_file, df_result)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)
