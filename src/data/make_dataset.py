import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'final.csv'
    df = pd.read_csv(data_path)

    # Impute all missing values in all the features
    df['price'].fillna(df['price'].median(), inplace=True)
    df['year_sold'].fillna(df['year_sold'].median(), inplace=True)
    df['property_tax'].fillna(df['property_tax'].median(), inplace=True)
    df['insurance'].fillna(df['insurance'].median(), inplace=True)
    df['beds'].fillna(df['beds'].median(), inplace=True)
    df['baths'].fillna(df['baths'].median(), inplace=True)
    df['sqft'].fillna(df['sqft'].median(), inplace=True)
    df['year_built'].fillna(df['year_built'].median(), inplace=True)
    df['lot_size'].fillna(df['lot_size'].median(), inplace=True)
    df['basement'].fillna(df['basement'].mode()[0], inplace=True)
    df['popular'].fillna(df['popular'].mode()[0], inplace=True)
    df['recession'].fillna(df['recession'].mode()[0], inplace=True)
    df['property_age'].fillna(df['property_age'].median(), inplace=True)
    df['property_type_Condo'].fillna(df['property_type_Condo'].mode()[0], inplace=True)

    # Drop duplicate rows from the data
    df = df.drop_duplicates()

    return df