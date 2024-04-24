import numpy as np
import pandas as pd
import data.external.wisconsin_bc as wbc 
from constants import MIN_ACCURACY 

from zenml import pipeline
from zenml.config import DockerSettings
from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from zenml.steps import BaseParameters, step_output
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.read_data import read_data
from data.raw.raw import get_spam_data_csv


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymenyTriggerParameters(BaseParameters):
  """Deployment trigger parameters"""
  min_accuracy: float = MIN_ACCURACY


@step 
def deployment_trigger(
  accuracy: float,
  config: DeploymenyTriggerParameters,
):
  """
  Implementation of the deployment trigger step
  """
  return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def deployment_pipeline(min_accuracy: float = MIN_ACCURACY, 
                        workers: int = 1, 
                        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
  # # Winsconsin Breast Cancer dataset
  # url = wbc.get_url()
  # names = wbc.get_names()

  # Spam dataset
  spam_data = get_spam_data_csv()
  encoding= 'iso8859_14'

  df = read_data(data_path=spam_data, encoding=encoding)
  X_train, X_test, y_train, y_test = clean_data(df)
  model = train_model(X_train, X_test, y_train, y_test)
  mse, r2, rmse, acc = evaluate_model(model, X_test, y_test)

  # Only deploy if the accuracy is greater than the min_accuracy
  should_deploy = deployment_trigger(accuracy=acc)


  mlflow_model_deployer_step(
    model=model,
    deploy_decision=should_deploy,
    workers=workers,
    timeout=timeout,
  )
