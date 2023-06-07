import pandas as pd
import re

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    return df

def to_csv(df, file_path):
    df.to_csv(file_path, index=True)
    return df

def to_xlsx(df, file_path):
    df.to_excel(file_path, index=False)
    return df
