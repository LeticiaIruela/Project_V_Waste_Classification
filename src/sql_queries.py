import os
import pymysql
import sqlalchemy as alch
import getpass
from dotenv import load_dotenv

def sql_connection(dbName):
    """
    Establishes a connection to a MySQL database using the provided database name.
    Args:
        dbName (str): The name of the MySQL database.
    Returns:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine object representing the database connection.
    """ 
    load_dotenv()
    password=os.getenv("password")
    connectionData=f"mysql+pymysql://root:{password}@localhost/{dbName}"
    engine = alch.create_engine(connectionData)

    return engine


def sql_exporting(df, dfName, dbName):
    """
    Exports a Pandas DataFrame to a specified table in a MySQL database.

    Args:
        df (pandas.DataFrame): The DataFrame to be exported.
        dfName (str): The name of the table to create or replace in the MySQL database.
        dbName (str): The name of the MySQL database to connect to.

    Returns:
        None
    """
    engine=sql_connection(dbName)
    df.to_sql(f"{dfName}", con=engine, if_exists='replace', index=False)
    