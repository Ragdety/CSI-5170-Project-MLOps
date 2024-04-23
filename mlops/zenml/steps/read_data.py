import logging
import pandas as pd

from zenml import step

class ReadData:
  """
  Data read class responsible for reading data from a given path
  """

  def __init__(self, data_path, names=None, encoding=None):
    """
    Initializes the ReadData class

    Args:
      data_path (str): path to the data
    """
    self.data_path = data_path
    self.names = names
    self.encoding = encoding

  def get_data(self):
    """
    Reads the CSV file provided in data_path and returns a Pandas DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file.
    """

    logging.info(f"Reading data from {self.data_path}")
    return pd.read_csv(self.data_path, names=self.names, encoding=self.encoding)
  

@step
def read_data(data_path: str, names=None, encoding=None) -> pd.DataFrame:
  """
  This step is responsible for reading a CSV file from the given data path
  and returning it as a Pandas DataFrame.

  Args:
      data_path (str): Path to the CSV file.

  Returns:
      pandas.DataFrame: DataFrame containing the data from the CSV file.

  Raises:
      Exception: If there is an error while ingesting the data.
  """
  try:
    read_data = ReadData(data_path, names, encoding)
    df = read_data.get_data()
    return df
  except Exception as e:
    logging.error(f"Error while reading data: {e}")
    raise e
  
