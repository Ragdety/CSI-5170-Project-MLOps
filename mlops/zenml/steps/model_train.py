import logging 
import sys 
import pandas as pd
from sklearn.base import RegressorMixin
from models.model_dev import LinearRegressionModel, RegularizedRegressionModel
from .config import ModelNameConfig

from zenml import step


@step
def train_model(
  X_train : pd.DataFrame, 
  X_test : pd.DataFrame, 
  y_train : pd.Series,
  y_test : pd.Series, 
  config : ModelNameConfig
)-> RegressorMixin:
  """
  Trains a machine learning model using the provided DataFrame.
  
  Args:
      df (pandas.DataFrame): DataFrame containing the training data.
  """

  model = None

  try:
    if config.model_name == "LogisticRegression":
      model = LinearRegressionModel()
      train_model = model.train(X_train, y_train)
      return train_model
  
    elif config.model_name == "RegularizedRegression":
      model = RegularizedRegressionModel()
      train_model = model.train()
      return train_model
    else:
      raise ValueError(f"Model {config.model_name} not supported.")

  except Exception as e:
    logging.error(f"Error while training the model: {e}")
    raise e
  