import logging 
import pandas as pd

from zenml import step
from data.strategy.data_cleaning import DataPreProcessingStrategy, DataSplitStrategy, DataCleaning
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"], 
]:
  """
  A step function to clean the data. It takes a DataFrame as input.

  Args:
      df (pandas.DataFrame): DataFrame containing the data to clean.

  Returns:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
  """
  try:
    process_strategy = DataPreProcessingStrategy()
    dc_process = DataCleaning(df, process_strategy)
    processed_data = dc_process.handle_data()

    # Note: processed_data is passed into the split strategy
    split_strategy = DataSplitStrategy()
    dc_split = DataCleaning(processed_data, split_strategy)
    X_train, X_test, y_train, y_test = dc_split.handle_data()

    logging.info(f"Data cleaned and split completed")
    return X_train, X_test, y_train, y_test
  except Exception as e:
    logging.error(f"Error while cleaning data: {e}")
    raise e
  
