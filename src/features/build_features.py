import pandas as pd

# create features
def build_features(df):
    
    # Store the processed dataset in data/processed
    df.to_csv('data/processed/processed_real_estate.csv', index=None)

    # Separate the input features and target variable
    X = df.drop('price', axis=1)
    y = df['price']

    return X, y