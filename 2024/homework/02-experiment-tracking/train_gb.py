import os
import pickle
import click
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_green_taxi_data")

from sklearn.ensemble import GradientBoostingRegressor
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
        mlflow.set_tag("estimator_name", 'gradient_boosting')
        mlflow.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        min_samples_split=20
        mlflow.log_param('min_samples_split', min_samples_split)
        gb = GradientBoostingRegressor(min_samples_split=min_samples_split)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_val)

        rmse_gb = root_mean_squared_error(y_val, y_pred_gb)
        mlflow.log_metric('rmse_gb', rmse_gb)
        mlflow.sklearn.log_model(gb, artifact_path='models_mlflow')


if __name__ == '__main__':
    run_train()
