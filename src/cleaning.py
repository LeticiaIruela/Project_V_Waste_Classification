import pandas as pd
import re

def clean_data(df):
    df.drop_duplicates(inplace=True)
    return df
