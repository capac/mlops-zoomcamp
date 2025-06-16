#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

home = Path.home()
data_dir = (home /
    'Programming/Python/data-talks-club/mlops-zoomcamp/2025/homework/04-deployment/data'
    )


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def predictions_by_year_and_month(year, month):
    data_file = data_dir / f"yellow_tripdata_{year}-0{month}.parquet"
    if data_file.exists():
        df = read_data(data_file)
    else:
        filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet"
        df = read_data(filename)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


# ## Solution to question #1
# 
# We'll start with the same notebook we ended up with in homework 1. We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2025/04-deployment/homework/starter.ipynb). Run this notebook for the March 2023 data. What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * **6.24**
# * 12.28
# * 18.28


# ## Solution to question #2
# 
# Like in the course videos, we want to prepare the dataframe with the output. First, let's create an artificial `ride_id` column:
# 
# `df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')`
# 
# Next, write the ride id and the predictions to a dataframe with results. Save it as parquet:
# ```
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * **66M**
# 
# **Note**: Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the dtypes of the columns and use pyarrow, not fastparquet.Like in the course videos, we want to prepare the dataframe with the output.


def df_result_by_year_and_month(year, month, output=False):
    
    data_file = data_dir / f"yellow_tripdata_{year}-0{month}.parquet"
    if data_file.exists():
        df = read_data(data_file)
    else:
        filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet"
        df = read_data(filename)

    dicts = df[categorical].to_dict(orient='records')

    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    y_pred_sr = pd.Series(y_pred, index=df.index)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.concat([df['ride_id'], y_pred_sr], axis=1)
    df_result.rename(columns={0: "predictions"}, inplace=True)

    if output:
        df_result.to_parquet('result.parquet', engine='pyarrow', compression=None, index=False)

    return df_result


# ## Solution to question #3
# 
# Now let's turn the notebook into a script. Which command you need to execute for that?
#
# jupyter nbconvert --to python notebook.ipynb


# Solution to question #4
#
# Now let's put everything into a virtual environment. We'll use pipenv for that. Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter notebook. After installing the libraries, pipenv creates two files: Pipfile and Pipfile.lock. The Pipfile.lock file keeps the hashes of the dependencies we use for the virtual env. What's the first hash for the Scikit-Learn dependency?
#
# The first hash for Scikit-Learn version 1.5.0 is `sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244c089afdebaad98264bff37c.`


# Solution to question #5
#
# Let's now make the script configurable via CLI. We'll create two parameters: year and month. Run the script for April 2023. What's the mean predicted duration?
#
#   7.29
#   **14.29**
#   21.29
#   28.29
#
# Hint: just add a print statement to your script.


if __name__ == "__main__":
    # solution to question #1
    year, month = (2023, 3)
    y_pred = predictions_by_year_and_month(year, month)
    print(f"The standard deviation of the predicted duration is {np.std(y_pred).round(2)}.")

    # solution to question #2
    _ = df_result_by_year_and_month(year, month, output=True)
    statinfo = os.stat("./result.parquet")
    print(f"File size is {round(statinfo.st_size/(1024*1024), 2)} MB.")

    # solution to question #5
    year, month = (2023, 4)
    df_result = df_result_by_year_and_month(year, month)
    print(f"Mean predicted duration is {np.mean(df_result.predictions).round(2)}.")
