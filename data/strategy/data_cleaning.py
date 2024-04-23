import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.preprocessing import LabelBinarizer
from typing import Union, Tuple
from data.strategy.spam_helpers import clean_text


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
      ############################################################
      # Wisconsin Breast Cancer dataset
      ############################################################

      # Adding dummy SVC model for now
      # Replace '?' with NaN and drop rows with missing values
      # df.replace('?', np.nan, inplace=True)
      # df.dropna(inplace=True)

      # # Convert 'Class' column to binary labels
      # df['Class'] = df['Class'].map(lambda x: 1 if x == 4 else 0)

      ############################################################
      # Spam dataset
      ############################################################
      df.drop(labels=df.columns[2:], axis=1, inplace=True)
      df.columns=['target', 'text']
      df['text'] = df['text'].apply(clean_text)

      return df
    except Exception as e:
      logging.error(f"Error while preprocessing data: {e}")
      raise e
    
class DataSplitStrategy(DataStrategy):
  """
  Class for data splitting into train and test
  """
  def handle_data(self, df: pd.DataFrame) -> Tuple[Union[pd.DataFrame, pd.Series], _VectorizerMixin]:
    """
    Split data
    """
    try:
      ############################################################
      # Wisconsin Breast Cancer dataset
      ############################################################
      # Split dataset into features and target variable
      # X = df.drop(['Sample code number', 'Class'], axis=1)
      # y = df['Class']

      ############################################################
      # SkLearn Breast Cancer dataset
      ############################################################
      # data = load_breast_cancer()
      # X, y = data.data, data.target
      # X, y = shuffle(X,y, random_state=16)

      # # Convert X and y to pandas DataFrame
      # X = pd.DataFrame(X, columns=data.feature_names)
      # y = pd.Series(y, name='Class')

      ############################################################
      # Spam dataset
      ############################################################

      # Create bag of words
      cv = CountVectorizer()
      X = cv.fit_transform(df['text']).toarray()
      X = pd.DataFrame(X, columns=cv.get_feature_names_out())

      # Target to binary
      lb = LabelBinarizer()
      y = lb.fit_transform(df['target']).ravel()
      y = pd.Series(y, name='target')

      return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), cv
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
