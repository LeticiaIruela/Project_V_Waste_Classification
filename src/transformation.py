import pandas as pd
import numpy as np
import os
import re 


def convert_INT(df, columns):
    df[columns] = df[columns].astype('int64')
    return df

def new_column(df, column_name, column_value):
    df[column_name] = column_value
    return df

def rename_columns(df, new_names):
    df = df.rename(columns=new_names)
    return df

def concat_dataframes(df1, df2, df3=None, df4=None, df5=None):
    dfs = [df1, df2]

    if df3 is not None:
        dfs.append(df3)
    if df4 is not None:
        dfs.append(df4)
    if df5 is not None:
        dfs.append(df5)

    df = pd.concat(dfs, axis=0)
    return df

def drop_columns(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop)
    return df

def modify_column_values(df, column, value_dict):
    df[column] = df[column].replace(value_dict)
    return df

def convert_INT(df, columns):
    df[columns] = df[columns].astype('int64')
    return df

def drop_columns_without_name(df):
    columns_without_name = df.columns[df.columns.map(lambda x: isinstance(x, float))]
    df = df.drop(columns_without_name, axis=1)
    return df

def rename_columns_and_drop_rows(df):
    rename_names = df.iloc[7].tolist()
    df.columns = rename_names
    df.drop(range(0, 9), axis=0, inplace=True)
    return df

def generated_rename_columns_and_drop_rows(df):
    rename_names = df.iloc[8].tolist()
    df.columns = rename_names
    df.drop(range(0, 10), axis=0, inplace=True)
    return df

def drop_columns_without_name(df):
    columns_without_name = df.columns[df.columns.map(lambda x: isinstance(x, float))]
    df = df.drop(columns_without_name, axis=1)
    return df

def drop_rows_with_colon(df):
    df = df[~df.astype(str).apply(lambda row: row.str.contains(":", na=False)).any(axis=1)]
    return df

def concat_dataframes(df1, df2, df3=None, df4=None, df5=None):
    dfs = [df1, df2]

    if df3 is not None:
        dfs.append(df3)
    if df4 is not None:
        dfs.append(df4)
    if df5 is not None:
        dfs.append(df5)

    df = pd.concat(dfs, axis=0)
    return df

def drop_rows_with_nan(df):
    df = df.dropna(axis=0, how='any')
    return df

def rename_columns_and_drop_rows(df):
    rename_names = df.iloc[7].tolist()
    df.columns = rename_names
    df.drop(range(0, 9), axis=0, inplace=True)
    return df

def drop_rows(df, column, values):
    df = df.dropna()
    df = df[~df[column].isin(values)]
    return df

def remove_columns(df, col):
    df = df.drop(columns=col)
    return df

def add_column_with_value(df, column_name, value):
    df[column_name] = value
    return df

def transform_month(df, month):
    df = df.melt(id_vars=['LOTE ', 'DISTRITO', 'FRACCIÓN'], value_vars=month, var_name='Month', value_name='Value')
    return df

#def transformar_meses(df, month):
    col_month = [col for col in df.columns if col in month]
    df = df.melt(id_vars=['LOTE ', 'DISTRITO', 'FRACCIÓN'], value_vars=col_month, var_name='Month', value_name='Value')
    return df

def process_dataframe(df):
    df["Month"] = df["Month"].apply(lambda x: x.split()[0].lower())
    df["Month"] = df["Month"].str.replace(r"\'\d{2}\(TN\)", "").str.lower()
    df["Month"] = df["Month"].str.split("'").str[0]
    return df

def col_conversion(df, new_order, col_to_remove=None):
    df = df[new_order]
    if col_to_remove is not None:
        df.drop(col_to_remove, axis=1, inplace=True)
    return df

def drop_rows(df, column, values):
    df = df.dropna()
    df = df[~df[column].isin(values)]
    return df

def rename_columns_and_drop_rows(df):
    rename_names = df.iloc[7].tolist()
    df.columns = rename_names
    df.drop(range(0, 9), axis=0, inplace=True)
    return df

def homogenize_district_names(df, district_column):
    def clean_district_name(value):
        value = re.sub(r'\d+\.\s*', '', value)
        value = value.capitalize()
        return value
    
    df = df.drop(df[(df[district_column].str.upper() == 'SIN DISTRITO') | (df[district_column].str.capitalize() == 'Sin distrito')].index)
    df[district_column] = df[district_column].apply(clean_district_name)
    return df

def rename_columns_and_add_country(df):
    rename_col = {
        'Código Interno del Situad': 'Internal Code',
        'Tipo Contenedor': 'Bin type',
        'Descripcion Modelo': 'Model',
        'DIRECCION': 'Address',
        'Cantidad':'Quantity',
        'Distrito':'Area'}
    df = df.rename(columns=rename_col)
    df['Country'] = 'Spain'
    df['City'] = 'Madrid'
    return df

def convert_index_to_column(df, column_name):
    df = df.reset_index()
    df = df.rename(columns={'index': column_name})
    return df

def process_dataframe1(df):
    # Obtener la primera fila como nombres de columnas y modificar los nombres existentes
    new_columns = df.iloc[0]
    new_columns = new_columns.replace({
        'Plásticos': 'Plastics',
        'Papel/Cartón*': 'Paper/cardboard',
        'Metales': 'Metalics',
        'Madera': 'Wood',
        'Total general': 'Total'
    })
    df.columns = new_columns
    
    # Eliminar la primera fila del DataFrame
    df = df.iloc[1:]
    
    # Convertir los valores a porcentaje y redondear a partir de la segunda columna
    df.iloc[:, 2:] = df.iloc[:, 2:].applymap(lambda x: f'{round(x * 100)}%')
    
    return df

