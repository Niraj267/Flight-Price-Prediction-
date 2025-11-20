
"""price_pipeline.py
A simple reusable pipeline example to load data, preprocess, and return ready X,y for training or inference.
"""
import pandas as pd
from src.preprocessing.clean_data import feature_engineer

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = feature_engineer(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y
