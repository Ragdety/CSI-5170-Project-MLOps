import logging 
import pandas as pd

from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
  """
  Trains a machine learning model using the provided DataFrame.
  
  Args:
      df (pandas.DataFrame): DataFrame containing the training data.
  """
  pass