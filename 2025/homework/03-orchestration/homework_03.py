#!/usr/bin/env python

from pathlib import Path
from prefect import flow, task
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

home = Path.home()
work_dir = home / "Programming/Python/data-talks-club/mlops-zoomcamp/2025/homework/03-orchestration"
data_dir = work_dir / "data"
data_file = data_dir / "yellow_tripdata_2023-03.parquet"


# ## Question 1. Select the Tool
# 
# You can use the same tool you used when completing the module, or choose a different one for your homework. What's the name of the orchestrator you chose?
# 
# ### Solution to problem #1
# **I'm using the Prefect orchestrator.**

# ## Question 2. Version
# What's the version of the orchestrator?
# 
# ### Solution to problem #2
# What is the version number of Prefect used?
# 
# **I'm using Prefect version 3.4.5.**

# ## Question 3. Creating a pipeline
# 
# Let's read the March 2023 Yellow taxi trips data. How many records did we load?
# 
# * 3,003,766
# * 3,203,766
# * **3,403,766**
# * 3,603,766
# 
# (Include a print statement in your code)


@task
def load_data(data):
    return pd.read_parquet(data)


# ## Question 4. Data preparation
# 
# Let's continue with pipeline creation. We will use the same logic for preparing the data we used previously. This is what we used (adjusted for yellow dataset):
# 
# ```
# def read_dataframe(filename):
#     df = pd.read_parquet(filename)
# 
#     df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
#     df.duration = df.duration.dt.total_seconds() / 60
# 
#     df = df[(df.duration >= 1) & (df.duration <= 60)]
# 
#     categorical = ['PULocationID', 'DOLocationID']
#     df[categorical] = df[categorical].astype(str)
#     
#     return df
# ```
# 
# Let's apply to the data we loaded in question 3. What's the size of the result?
# 
# * 2,903,766
# * 3,103,766
# * **3,316,216**
# * 3,503,766


@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


# ## Question 5. Train a model
# 
# We will now train a linear regression model using the same code as in homework 1.
# 
# * Fit a dict vectorizer.
# * Train a linear regression with default parameters.
# * Use pick up and drop off locations separately, don't create a combination feature.
# 
# Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model.
# 
# What's the intercept of the model?
# 
# Hint: print the `intercept_` field in the code block
# 
# * 21.77
# * **24.77**
# * 27.77
# * 31.77


@flow(log_prints=True)
def transform_dataframe(df):
    dv = DictVectorizer()
    pu_do_loc_df = df[['PULocationID', 'DOLocationID']].astype(str)
    pu_do_loc_dict = pu_do_loc_df.to_dict(orient='records')
    pu_do_loc_arr = dv.fit_transform(pu_do_loc_dict)

    X = pu_do_loc_arr.copy()
    y = df['duration'].copy()

    lr = LinearRegression()
    lr.fit(X, y)
    
    return lr.intercept_


# ## Question 6. Register the model
# 
# The model is trained, so let's save it with MLFlow. Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field):
# 
# * 14,534
# * 9,534
# * **4,534**
# * 1,534

# In[11]:


EXPERIMENT_NAME = "linear-regression-models"
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()


@task(log_prints=True)
def transform_dataframe(df):
    with mlflow.start_run():
        dv = DictVectorizer()
        pu_do_loc_df = df[['PULocationID', 'DOLocationID']].astype(str)
        pu_do_loc_dict = pu_do_loc_df.to_dict(orient='records')
        pu_do_loc_arr = dv.fit_transform(pu_do_loc_dict)

        X = pu_do_loc_arr.copy()
        y = df['duration'].copy()

        lr = LinearRegression()
        lr.fit(X, y)
    
        return lr.intercept_


@flow(log_prints=True)
def register_model(df):

    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
        )
    
    # Register the model
    print(f"run id: {runs[0].info.run_id}")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(model_uri)

    mlflow.register_model(model_uri, name="yellow-taxi-linear-regressor")


def main():
    df = load_data(data_file)
    print(f"The number of files loaded by Prefect is {df.shape[0]}.")
    yellow_taxi_df = read_dataframe(data_file)
    print(f"The size of the dataframe loaded by Prefect is {yellow_taxi_df.shape[0]}.")
    y_intercept = transform_dataframe(yellow_taxi_df)
    print(f"The y-intercept field is {np.round(y_intercept, 5)}")
    register_model(yellow_taxi_df)


if __name__ == "__main__":
    main()
