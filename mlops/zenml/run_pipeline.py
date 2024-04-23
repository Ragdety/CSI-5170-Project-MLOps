import data.external.wisconsin_bc as wbc
import os

from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
from data.raw.raw import get_spam_data_csv


def main():
  # Winsconsin Breast Cancer dataset
  url = wbc.get_url()
  column_names = wbc.get_names()

  # Spam dataset
  spam_data = get_spam_data_csv()
  encoding= 'iso8859_14'

  print("Tracking URI: ")
  print(Client().active_stack.experiment_tracker.get_tracking_uri())

  # training_pipeline(data_path=url, names=column_names)
  training_pipeline(data_path=spam_data, encoding=encoding)


if __name__ == "__main__":
  main()
