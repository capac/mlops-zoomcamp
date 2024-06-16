#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sys import argv

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(data_link):
    print('Downloading data...')
    df = pd.read_parquet(data_link)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print('Returning dataframe...')
    return df


def calculate_prediction(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print('Calculating predictions...')
    return y_pred


def main():
    year, month = argv[1], argv[2]
    data_link = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    df = read_data(data_link)
    y_pred = calculate_prediction(df)
    print(f'The mean prediction is {round(np.mean(y_pred), 2)}.')


if __name__ == "__main__":
    main()
