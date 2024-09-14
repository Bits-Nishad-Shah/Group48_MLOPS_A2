from dataprep.clean import clean_headers
import pandas as pd
import os

def generate_directories():
    """
    Create directories
    """
    directories_to_create = [
        "/artifacts/models/reports/eda/",
        "/artifacts/models/reports/shap/",
        "/artifacts/models/models/",
        "/datasets/processed/"
    ]
    for path in directories_to_create:
        os.makedirs(os.getcwd() + path, exist_ok=True)


def get_raw_data():
    """
    Read and return train and test data
    """
    # Load the dataset
    df_train = pd.read_csv(os.getcwd() + '/datasets/raw/train.csv')
    df_test = pd.read_csv(os.getcwd() + '/datasets/raw/test.csv') 
    return df_train, df_test

def clean_df_headers(df):
    """
    Converts column headers to snake case
    """
    # Cleaning the headers
    return clean_headers(df, case='snake')

