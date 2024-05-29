import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

runs = client.search_runs(
    experiment_ids='1',
    filter_string='metrics.rmse < 6.0',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=['metrics.rmse ASC'],
)

for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")

run_id = "0eb07d88628d429b87176fec29fb7aa1"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, name="nyc-taxi-regressor")
