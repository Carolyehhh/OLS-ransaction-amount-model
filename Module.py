import statsmodels.api as sm
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from data_SQLquery_list import MacroData

def Connect_to_MSSQL():
    """
    連接 SQL Server
    """
    try: 
        print("GSQLAlchemy 連接 MSSQL 資料庫")
        engine = create_engine('mssql+pymssql://carol_yeh:Cmoneywork1102@192.168.121.50/master')
        commection = engine.connect()
        print("GSQLAlchemy 連接.success")
        return engine
    except Exception as e:
        print(f"An error occured: {e}")
        return None
    finally:
        pass

def extract_data(data_SQLquery_list):
    """
    return all_data : list
    """
    engine = Connect_to_MSSQL()
    if engine is not None:
        all_data = []
        for query in data_SQLquery_list:
            try:
                read_data = pd.read_sql(query, engine)
                all_data.append(read_data)
            except Exception as e:
                print(f"Error when extracting data: {e}")
            engine.dispose()

    else:
        print("Failed to connect to the database")
        return None
    return all_data

def get_last_month_of_quarter(year_month):
    year = int(year_month[:4])
    month = int(year_month[-2:])
    if month in [1, 2, 3]:
        return f"{year}03"
    elif month in [4, 5, 6]:
        return f"{year}06"
    elif month in [7, 8, 9]:
        return f"{year}09"
    elif month in [10, 11, 12]:
        return f"{year}12"



