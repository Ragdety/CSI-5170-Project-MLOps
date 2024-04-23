import logging 
import pandas as pd
import mlflow

from sklearn.base import RegressorMixin, BaseEstimator
from models.model_dev import (
  LinearRegressionModel,
  MyLogisticRegressionModel,
  MyNaiveBayesModel, 
  NaiveBayesModel,
  LogisticRegressionModel
)
from .config import ModelNameConfig
from zenml import step
from zenml.client import Client
from constants import *


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
  X_train : pd.DataFrame, 
  X_test : pd.DataFrame, 
  y_train : pd.Series,
  y_test : pd.Series, 
  config : ModelNameConfig
)-> BaseEstimator:
  """
  Trains a machine learning model using the provided DataFrame.
  
  Args:
      df (pandas.DataFrame): DataFrame containing the training data.
  """

  model = None

  logging.info(f"Training model: {config.model_name}")

  try:
    # Automatically log models, scores, etc...
    mlflow.sklearn.autolog()
    if config.model_name == LINEAR_REGRESSION:
      model = LinearRegressionModel()
      train_model = model.train(X_train, y_train)
      return train_model
    
    elif config.model_name == LOGISTIC_REGRESSION:
      model = LogisticRegressionModel()
      train_model = model.train(X_train, y_train)
      return train_model

    elif config.model_name == MY_LOGISTIC_REGRESSION:
      model = MyLogisticRegressionModel()
      train_model = model.train(X_train, y_train)
      return train_model
    
    elif config.model_name == NAIVE_BAYES:
      model = NaiveBayesModel()
      train_model = model.train(X_train, y_train)
      return train_model
    
    elif config.model_name == MY_NAIVE_BAYES:
      model = MyNaiveBayesModel()
      train_model = model.train(X_train, y_train)
      return train_model

    else:
      raise ValueError(f"Model {config.model_name} not supported.")

  except Exception as e:
    logging.error(f"Error while training the model: {e}")
    raise e
  