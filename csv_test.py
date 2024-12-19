# Import the PPI.csv file
import csv
import pandas as pd

# 讀取 CSV 並轉換為 DataFrame
PPI_df = pd.read_csv('PPI_TW.csv', usecols=[0, 1], header=None, nrows=47, skiprows=1)
PPI_df.columns = ['original_date', 'overall_ppi']

# 日期轉換函式
def convert_date (original_date):
    year = int(original_date[:3]) + 1911
    month_part = original_date.split('~')[1] # Extract the part after '~'
    month = ''.join(filter(str.isdigit, month_part))
    return f"{year}{int(month):02d}" # Format as YYYYMM

PPI_df['年月'] = PPI_df['original_date'].apply(convert_date)
PPI_df = PPI_df[['年月', 'overall_ppi']]

# print(PPI_df)