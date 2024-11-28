# Goal:大盤成交金額預估
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from Module import Connect_to_MSSQL, extract_data
from data_SQLquery_list import MacroData

# Extract Macro data
data = extract_data(MacroData)
data = [pd.DataFrame(element) for element in data]
# print(data[0].loc[data[0]['年月']=='202201'])

# 先對月-總經資料做轉置
pivot_MacroVar = data[0].pivot(index='年月', columns='名稱', values='數值')
# print(pivot_MacroVar)

# Combine Data into DataFrame
Combined_df = pd.merge(data[2], pivot_MacroVar,how='left', on=['年月','年月'])
# print(Combined_df[Combined_df['年月']=='202411'])
print(Combined_df.columns)

# 好像沒撈到M2%!!!!、YY:農曆年、L1:成交金額(億)-上個月=>成交金額(千)_月
Combined_df = Combined_df[['年月', '經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元', '平均每人國內生產毛額(GDP)–美元', '躉售物價指數', '躉售物價指數年增率', '消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均', '中央銀行重貼現率', '台灣失業率(經季節性調整)']]

print(Combined_df)
# print(data[0].columns)
# print(data[2].columns)

# 確認回歸模型的前提條件
# Check



# #duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
# #Y = duncan_prestige.data['income']
# #X = duncan_prestige.data['education']
# #x = sm.add_constant(X)

# df=pd.read_excel("成交金額預估.xlsx",sheet_name="model_4")
# Y=df["成交億"]
# X=df.iloc[:,2:23]
# X=pd.DataFrame(df,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',"q1","q2","q3","q4",'GDP%',"GDP","人均GDP"])
# X=pd.DataFrame(df,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',
#                          "人均GDP","M1b%","L1","市場",])
# X=pd.DataFrame(df,columns=["人均GDP","CPI","M1b%","L1","市場","失業"])

# model = sm.OLS(Y,X)
# result = model.fit()
# print(result.summary())