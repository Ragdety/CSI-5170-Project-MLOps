import click

from mlops.zenml.pipelines.deployment_pipeline import deployment_pipeline
from constants import *
from rich import print
from typing import cast

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.client import Client


@click.command()
@click.option(
  "--config",
  "-c",
  type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
  default=DEPLOY_AND_PREDICT,
  help="Choose the configuration to run between " \
    "DEPLOY, PREDICT, DEPLOY_AND_PREDICT. " \
    "Default is DEPLOY_AND_PREDICT",
)

@click.option(
  "--min-accuracy",
  default=0.8,
  help="Minimum accuracy required for the model to be deployed",
)

def run_deployment(config: str, min_accuracy: float):
  mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

  deploy_and_predict = config == DEPLOY_AND_PREDICT

  # Deploy if the config is DEPLOY or DEPLOY_AND_PREDICT
  deploy = config == DEPLOY or deploy_and_predict

  # Predict if the config is PREDICT or DEPLOY_AND_PREDICT
  predict = config == PREDICT or deploy_and_predict

  print("Tracking URI: ")
  print(Client().active_stack.experiment_tracker.get_tracking_uri())

  if deploy:
    deployment_pipeline(
      min_accuracy=min_accuracy,
      workers=3,
      timeout=600  
    )
  # if predict:
  #   inference_pipeline()

  print(
    "You can run:"
    f"[italic green] mlflow ui --backend-store-uri {get_tracking_uri()}[/italic green]\n"
    "to inspect your experiment runs within the MLflow UI.\n"
  )

  existing_services = mlflow_model_deployer_component.find_model_server(
    pipeline_name="deployment_pipeline",
    pipeline_step_name="mlflow_model_deployer_step",
    model_name="model",
  )

  if existing_services:
    service = cast(MLFlowDeploymentService, existing_services[0])

    if service.is_running:
      print(
        "The model server is running locally as a daemon "
        "process service and accepts requests at: "
        f"[bold]{service.prediction_url}[/bold]\n"
        "To stop the service, run: "
        f"[italic green]zenml model-deployer models delete {str(service.uuid)}[/italic green]." 
      )
    elif service.is_failed:
      print(
        "The MLFlow prediction server is in a failed state: \n"
        f"[bold]Last state: {service.status.state.value}[/bold]"
        f"[bold]Last error: {service.status.last_error}[/bold]"
      )

  else:
    print(
      "No MLflow prediction server is running. "
      "The deploymeny pipeline must run first to train a model and deploy it. "
      "Execute the same command with the --deploy argument to deploy a model."
    )

if __name__ == "__main__":
  run_deployment()