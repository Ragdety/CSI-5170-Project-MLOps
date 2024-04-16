from pipelines.training_pipeline import training_pipeline


def main():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

  column_names = [
      "Sample code number",
      "Clump Thickness",
      "Uniformity of Cell Size",
      "Uniformity of Cell Shape",
      "Marginal Adhesion",
      "Single Epithelial Cell Size",
      "Bare Nuclei",
      "Bland Chromatin",
      "Normal Nucleoli",
      "Mitoses",
      "Class"
  ]

  training_pipeline(data_path=url, names=column_names)


if __name__ == "__main__":
  main()