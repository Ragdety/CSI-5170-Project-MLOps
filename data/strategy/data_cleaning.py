import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
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

      # Adding dummy SVC model for now
      # Replace '?' with NaN and drop rows with missing values
      df.replace('?', np.nan, inplace=True)
      df.dropna(inplace=True)

      # Convert 'Class' column to binary labels
      df['Class'] = df['Class'].map(lambda x: 1 if x == 4 else 0)
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
      # Split dataset into features and target variable
      X = df.drop(['Sample code number', 'Class'], axis=1)
      y = df['Class']
      # data = load_breast_cancer()
      # X, y = data.data, data.target

      # X,y = shuffle(X,y, random_state=16)

      return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
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
