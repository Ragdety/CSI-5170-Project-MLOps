from zenml import pipeline
from steps.read_data import read_data
from steps.model_train import train_model
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model


@pipeline
def training_pipeline(data_path: str):
  """
  Training pipeline
  """
  df = read_data(data_path)
  clean_data(df)
  train_model(df)
  evaluate_model(df)

