import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Union


class DataStrategy(ABC):
  """
  Abstract class defining strategy for data cleaning
  """

  @abstractmethod
  def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    pass


class DataPreProcessingStrategy(DataStrategy):
  """
  Class for data preprocessing
  """
  def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    """
    Preprocess data
    """
    try:
      # TODO: Add our data preprocessing logic here
      df = df.dropna()
      return df
    except Exception as e:
      logging.error(f"Error while preprocessing data: {e}")
      raise e
    
class DataSplitStrategy(DataStrategy):
  """
  Class for data splitting into train and test
  """
  def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    """
    Split data
    """
    try:
      # TODO: Add our data splitting logic here
      X = df.iloc[:, :-1]
      y = df.iloc[:, -1]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      return X_train, X_test, y_train, y_test
      # return df
    except Exception as e:
      logging.error(f"Error while splitting data: {e}")
      raise e

class DataCleaning:
  """
  Class for data cleaning
  """
  def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
    self.data = data
    self.strategy = strategy

  def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
    """
    Clean data
    """
    try:
      return self.strategy.handle_data(self.data)
    except Exception as e:
      logging.error(f"Error while cleaning data: {e}")
      raise e
