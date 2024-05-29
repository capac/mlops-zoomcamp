import os
import pickle
import click
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_green_taxi_data")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    with mlflow.start_run():
        mlflow.set_tag("developer", 'angelo')
        # mlflow.set_tag("model", 'random_forest')
        mlflow.set_tag("estimator_name", 'random_forest')
        mlflow.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        max_depth=10
        mlflow.log_param('max_depth', max_depth)
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_val)

        rmse_rf = root_mean_squared_error(y_val, y_pred_rf)
        mlflow.log_metric('rmse_rf', rmse_rf)
        mlflow.sklearn.log_model(rf, artifact_path='models_mlflow')


if __name__ == '__main__':
    run_train()
