# Goal:大盤成交金額預估，目標是用前一期的數值預估下一期的「市場成交金額」
# 資料期間：201701-202411
# 2023年前使用 WPI，2023年後使用 PPI >> PPI使用「中華民國統計資訊網」的資料
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor # 檢查共線性(VIF)
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Ridge # 正則化
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
import csv
from Module import Connect_to_MSSQL, extract_data, get_last_month_of_quarter
from data_SQLquery_list import MacroData

# Extract Macro data
data = extract_data(MacroData)
data = [pd.DataFrame(element) for element in data]
# print(data[3]) #農曆年月份 #print(data[4]) #交易天數

# 讀取 PPI 的 CSV 檔案並轉換為 DataFrame
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

# 先對月-總經資料做轉置
pivot_MacroVar = data[0].pivot(index='年月', columns='名稱', values='數值')

# Combine Data into DataFrame
Combined_df = pd.merge(data[2], pivot_MacroVar,how='left', on=['年月','年月'])

end_date = '202411' # 定義變數

# Filter, merge and process data
Combined_df = (
    pd.merge(
        Combined_df.query(f"年月 >= '201601' and 年月 <= '{end_date}'")[
            ['年月', '成交金額(千)_月', '消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均', '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', '躉售物價指數', '躉售物價指數年增率']
        ],
        data[3],
        how='left',
        on = ['年月','年月']
    )[
        ['年月', '成交金額(千)_月', '是否放假', '消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均', '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', '躉售物價指數', '躉售物價指數年增率']
    ]

)

# L1:成交金額(億)-上個月=>成交金額(千)_月、L2:成交金額(億)-上上個月
Combined_df['上個月成交金額(千)_月'] = Combined_df['成交金額(千)_月'].shift(1)
Combined_df['上上個月成交金額(千)_月'] = Combined_df['成交金額(千)_月'].shift(2)

# 取201701後的資料(配合現有的農曆年資料)、data[4] = 交易天數
Combined_df = (
    pd.merge(Combined_df.query(f"年月 >= '201701' and 年月 <= '{end_date}'"), data[4], how='left', on=['年月', '年月'])
)
# 春節是否放假，非春節月份顯示'0'
Combined_df['是否放假'] = Combined_df['是否放假'].fillna(0)
# print(Combined_df)
Combined_df['年季'] = Combined_df['年月'].apply(get_last_month_of_quarter)

# 檢查pivot_MacroVar並重設索引
pivot_MacroVar.reset_index(inplace=True)

pivot_MacroVar['年季'] = pivot_MacroVar['年月'].apply(get_last_month_of_quarter)
Combined_df = pd.merge(Combined_df, pivot_MacroVar[['年季','經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元', '平均每人國內生產毛額(GDP)–美元']].dropna(), how='left', on=['年季', '年季'])
# print(len(Combined_df)) #94, 20

# left join PPI
Combined_df = pd.merge(Combined_df, PPI_df, how='left', on='年月')

# 加入PPI
Combined_df['年分'] = Combined_df['年月'].str[:4]
Combined_df['生產者物價指數'] = Combined_df.apply(
    lambda row: row['躉售物價指數'] if int(row['年分']) < 2023 else row['overall_ppi'], axis=1
)

Combined_df = Combined_df.drop(['躉售物價指數', '躉售物價指數年增率', '年季', 'overall_ppi', '年分'], axis=1)

# 變數轉換（如對數轉換）
Combined_df[['log_生產者物價指數', 'log_交易天數_月', 'log_平均每人國內生產毛額(GDP)–美元', 'log_貨幣總額年增率(M1B)期底', 'log_貨幣總額年增率(M2)期底']] = Combined_df[['生產者物價指數', '交易天數_月', '平均每人國內生產毛額(GDP)–美元', '貨幣總額年增率(M1B)期底', '貨幣總額年增率(M2)期底']].apply(np.log)

# 更新 X 變數集
# 刪除高共線性+高相關性的變數
# 先刪除M2(留下M1B)、刪除CPI年增率(留下CPI)、刪除國內生產毛額(GDP)–美元(留下-平均每人國內生產毛額(GDP)–美元)、刪除消費者物價指數(CPI)
# 刪除貨幣總額(M1B)日平均(與貨幣總額(M2)日平均 相關性極高，並且 VIF 值過高)
# 刪除log_平均每人國內生產毛額(GDP)–美元(VIF大且Ridge貢獻度小)
X3 = Combined_df[['是否放假',
       'log_貨幣總額年增率(M1B)期底', 'log_貨幣總額年增率(M2)期底',
       '中央銀行重貼現率', '失業率', '上個月成交金額(千)_月', '上上個月成交金額(千)_月', 'log_交易天數_月',
       '經濟成長率(GDP)–單季', 'log_生產者物價指數']]
X3 = sm.add_constant(X3)

# # 重新計算 VIF
# vif3 = pd.DataFrame()
# vif3['變數'] = X3.columns
# vif3['VIF'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
# print(vif3)

# # 檢查相關性
# correlattion_matrix = Combined_df[['是否放假',
#        'log_貨幣總額年增率(M1B)期底', 'log_貨幣總額年增率(M2)期底',
#        '中央銀行重貼現率', '失業率', '上個月成交金額(千)_月', '上上個月成交金額(千)_月', 'log_交易天數_月',
#        '經濟成長率(GDP)–單季', 'log_生產者物價指數']].corr()
# print(correlattion_matrix)

#-----------------------------------------------------------------------------------
# 排除"年月"和"成交金額(千)_月"，對其餘變數進行滯後處理
last_row = Combined_df[Combined_df['年月'] == '202411']
new_row = last_row.copy()
new_row['年月'] = '202412'

lagged_columns = ['是否放假', 'log_貨幣總額年增率(M1B)期底', 'log_貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', 
                  '上個月成交金額(千)_月', '上上個月成交金額(千)_月', 'log_交易天數_月','經濟成長率(GDP)–單季', 'log_生產者物價指數']

# 將 202412 的 Lag 特徵填入
for col in lagged_columns:
    new_row[f'Lag_{col}'] = last_row[col].values[0]

# 合併到原始 DataFrame
Combined_df = pd.concat([Combined_df, new_row], ignore_index = True)

# 創建滯後變數 (Lag Variables)
for col in lagged_columns:
    Combined_df[f'Lag_{col}'] = Combined_df[col].shift(1)

if Combined_df[Combined_df['年月'] == '202411'].empty:
    print("日期 '202411' 不存在於 DataFrame 中，請檢查資料。")
else:
    Combined_df.loc[Combined_df['年月'] == '202411', '失業率'] = 3.36

# na_rows = Combined_df[Combined_df.isna().any(axis=1)]
# print(na_rows)

# 由於滯後變數會讓第一期缺失，可以去掉缺失值的行
Combined_lag = Combined_df.copy().dropna()

# 設置X和Y變數
X_lag = Combined_lag[['Lag_' + col for col in lagged_columns]]
Y_lag = Combined_lag['成交金額(千)_月']

""""
TO DO:
沒有使用11月的數值預測12月的金額
"""
train_data = Combined_lag[Combined_lag['年月'] < '202412']
test_data = Combined_lag[Combined_lag['年月'] == '202412']
feature_columns = [
    'Lag_是否放假', 
    'Lag_log_貨幣總額年增率(M1B)期底', 
    'Lag_log_貨幣總額年增率(M2)期底', 
    'Lag_是否放假', 'Lag_失業率', 
    'Lag_上個月成交金額(千)_月', 
    'Lag_上上個月成交金額(千)_月', 
    'Lag_log_交易天數_月', 
    'Lag_經濟成長率(GDP)–單季', 
    'Lag_log_生產者物價指數'
]

if 'const' in feature_columns:
    feature_columns.remove('const')

X_train = train_data[feature_columns]
Y_train = train_data['成交金額(千)_月']
X_test = test_data[feature_columns]
# print(X_test.columns)

# add constant 
X_train = sm.add_constant(X_train, has_constant='add')
X_test = sm.add_constant(X_test, has_constant='add')
# print(X_test.columns)

# OLS model 
model_lag = sm.OLS(Y_train, X_train)
result_lag = model_lag.fit()

Y_pred = result_lag.predict(X_test)
print(f"預測的市場成交金額 (千): {Y_pred.values[0]:,.2f}") # 預測的市場成交金額 (千): 5,917,266,043.80
# print(Combrined_lag[['log_貨幣總額年增率(M1B)期底', 'Lag_log_貨幣總額年增率(M1B)期底']])


