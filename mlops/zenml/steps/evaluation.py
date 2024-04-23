import logging 
import pandas as pd
import mlflow

from zenml import step
from models.evaluation import MSE, R2, RMSE, Accuracy
from sklearn.base import RegressorMixin, BaseEstimator
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


@step
def evaluate_model(
  model: BaseEstimator, 
  X_test: pd.DataFrame, 
  y_test: pd.DataFrame
) -> Tuple[
  Annotated[float, "MSE"],
  Annotated[float, "R2"],
  Annotated[float, "RMSE"],
  Annotated[float, "Accuracy"]
]:
  """
  A step function to evaluate the model using the provided DataFrame.
  
  Args:
      df (pandas.DataFrame): DataFrame containing the data to evaluate.
  """
  try:
    prediction = model.predict(X_test)

    mse_obj = MSE()
    mse = mse_obj.calculate_scores(y_test, prediction)
    mlflow.log_metric("MSE", mse)

    r2_obj = R2()
    r2 = r2_obj.calculate_scores(y_test, prediction)
    mlflow.log_metric("R2", r2)

    rmse_obj = RMSE()
    rmse = rmse_obj.calculate_scores(y_test, prediction)
    mlflow.log_metric("RMSE", rmse)

    acc_obj = Accuracy()
    acc = acc_obj.calculate_scores(y_test, prediction)
    mlflow.log_metric("Accuracy", acc)

    logging.info(f"Model evaluation completed!")

    return mse, r2, rmse, acc
  except Exception as e:
    logging.error(f"Error while evaluating the model: {e}")
    raise e
