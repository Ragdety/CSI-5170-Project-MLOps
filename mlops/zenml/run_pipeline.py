from pipelines.training_pipeline import training_pipeline


def main():
  training_pipeline(data_path="../../data/raw/data.csv")


if __name__ == "__main__":
  main()