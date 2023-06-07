import pandas as pd
import re

def pd_excel(file_path):
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        dfs = {}
        for sheet_name in sheet_names:
            df = excel_file.parse(sheet_name)
            dfs[sheet_name] = df

        collected_yellow_bin = dfs.get('BBDD Amarillo')
        collected_blue_bin = dfs.get('BBDD Azul')
        SP_bins = dfs.get('Contenedores')
        return collected_yellow_bin, collected_blue_bin, SP_bins
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file '{file_path}': {str(e)}")
        return None
    
def read_excel_sheets(file_path):
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    dfs = {}
    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)
        dfs[sheet_name] = df
    return dfs

def read_excel_to_df(file_path):
    df = pd.read_excel(file_path)
    return df

def process_dataframe(df):
    new_columns = df.iloc[0]
    new_columns = new_columns.replace({
        'Plásticos': 'Plastics',
        'Papel/Cartón*': 'Paper/cardboard',
        'Metales': 'Metalics',
        'Madera': 'Wood',
        'Total general': 'Total'
    })
    df.columns = new_columns
    df = df.iloc[1:]
    for col in df.columns[1:]:
        df[col] = (df[col] * 100).astype(int).astype(str) + '%'
    return df


