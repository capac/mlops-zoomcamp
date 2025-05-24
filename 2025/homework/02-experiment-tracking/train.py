import os
import pickle
import click
import mlflow
from mlflow import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('nyc-green-taxi-data-2023')


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.autolog()
    with mlflow.start_run() as run:

        mlflow.set_tag('developer', 'angelo')
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # min_samples_split = 10
        # mlflow.log_param('min_samples_split', min_samples_split)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)
        
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == '__main__':
    run_train()
