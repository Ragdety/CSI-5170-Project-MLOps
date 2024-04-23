import pandas as pd
import os

data_db = {
    "spam": "../../../data/raw/spam.csv",
    "wisconsin": "../../../data/raw/data.csv"
}

def get_spam_data_df() -> pd.DataFrame:
    return pd.read_csv(data_db["spam"])

def get_wisconsin_data_df() -> pd.DataFrame:
    return pd.read_csv(data_db["wisconsin"])

def get_spam_data_csv() -> str:
    return data_db["spam"]

def get_wisconsin_data_csv() -> str:
    return data_db["wisconsin"]