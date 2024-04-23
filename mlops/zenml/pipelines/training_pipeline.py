from zenml import pipeline
from steps.read_data import read_data
from steps.model_train import train_model
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model


@pipeline
def training_pipeline(data_path: str, names=None, encoding=None):
  """
  Training pipeline
  """
  df = read_data(data_path, names=names, encoding=encoding)
  X_train, X_test, y_train, y_test, vectorizer = clean_data(df)
  model = train_model(X_train, X_test, y_train, y_test)
  mse, r2, rmse, acc = evaluate_model(model, X_test, y_test)


