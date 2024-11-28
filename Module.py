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

# test = extract_data(MacroData)
# print(test)



