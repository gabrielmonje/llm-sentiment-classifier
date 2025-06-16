import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_text(df, text_column):
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].str.strip()
    return df
