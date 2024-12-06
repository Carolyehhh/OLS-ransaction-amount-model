# Goal:大盤成交金額預估，資料從201701-202410(因為202411的部分資料，如:CPI、M1B尚未公布)、相比於建文的檔案排除WPI相關數值，因為「主計總處自資料時間112年1月起停編躉售物價指數」
# WPI 改使用 PPI
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from Module import Connect_to_MSSQL, extract_data, get_last_month_of_quarter
from data_SQLquery_list import MacroData

# Extract Macro data
data = extract_data(MacroData)
data = [pd.DataFrame(element) for element in data]
# print(data[3]) #農曆年月份 #print(data[4]) #交易天數

# 先對月-總經資料做轉置
pivot_MacroVar = data[0].pivot(index='年月', columns='名稱', values='數值')

# Combine Data into DataFrame
Combined_df = pd.merge(data[2], pivot_MacroVar,how='left', on=['年月','年月'])

# Filter, merge and process data
Combined_df = (
    pd.merge(
        Combined_df.query("年月 >= '201601' and 年月 <= '202410'")[
            ['年月', '成交金額(千)_月', '消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均', '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率']
        ],
        data[3],
        how='left',
        on = ['年月','年月']
    )[
        ['年月', '成交金額(千)_月', '是否放假', '消費者物價指數(CPI)', '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均', '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率']
    ]

)

# L1:成交金額(億)-上個月=>成交金額(千)_月、L2:成交金額(億)-上上個月
Combined_df['上個月成交金額(千)_月'] = Combined_df['成交金額(千)_月'].shift(1)
Combined_df['上上個月成交金額(千)_月'] = Combined_df['成交金額(千)_月'].shift(2)

# 取201701後的資料(配合現有的農曆年資料)、data[4] = 交易天數
Combined_df = (
    pd.merge(Combined_df.query("年月 >= '201701' and 年月 <= '202410'"), data[4], how='left', on=['年月', '年月'])
)

# 春節是否放假，非春節月份顯示'0'
Combined_df['是否放假'] = Combined_df['是否放假'].fillna(0)
# print(Combined_df)
Combined_df['年季'] = Combined_df['年月'].apply(get_last_month_of_quarter)

# 檢查pivot_MacroVar並重設索引
pivot_MacroVar.reset_index(inplace=True)

pivot_MacroVar['年季'] = pivot_MacroVar['年月'].apply(get_last_month_of_quarter)
Combined_df = pd.merge(Combined_df, pivot_MacroVar[['年季','經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元', '平均每人國內生產毛額(GDP)–美元']].dropna(), how='left', on=['年季', '年季'])

# 確認回歸模型的前提條件
# 母體資料的線性關係
# 相關性
X = Combined_df[['是否放假', '經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元',
       '平均每人國內生產毛額(GDP)–美元', '消費者物價指數(CPI)',
       '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均',
       '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', '交易天數_月', '上個月成交金額(千)_月',
       '上上個月成交金額(千)_月']]
Y = Combined_df['成交金額(千)_月']
correlation_matrix = X.corrwith(Y)
# print(correlation_matrix)

# OLS Model
transaction_amount_model0 = sm.OLS(Y, X)
result0 = transaction_amount_model0.fit()
print(result0.summary())

# 殘差項
# A.常態分佈
# sns.histplot(residuals, kde=True) # kde=True 繪製核密度曲線
# plt.show()
# B.變異數同質性
# C.各自獨立





# =================
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

# ========================================
# 散佈圖備用QQ (size不一致)
# # scatter plot
# X = Combined_df[['是否放假', '經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元',
#        '平均每人國內生產毛額(GDP)–美元', '消費者物價指數(CPI)',
#        '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底', '貨幣總額(M2)日平均',
#        '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', '交易天數_月', '上個月成交金額(千)_月',
#        '上上個月成交金額(千)_月']]
# Y = Combined_df['成交金額(千)_月']

# # 找出NULL值
# # print(X.isnull().sum())
# # print(Y.isnull().sum())
# # print(X.dtypes) 
# # print(Y.dtypes)
# X = X.apply(pd.to_numeric, errors='coerce')
# X = X.dropna().reset_index(drop=True)
# Y = Y.loc[X.index].reset_index(drop=True)
# # print(X.dtypes) # Objects
# # print(Y.dtypes)

# # print(Combined_df.shape)
# # print(X.shape)
# # print(Y.shape)

# plt.scatter(X, Y)

# # Add Title
# # plt.title('散佈圖')
# # plt.xlabel('X variables')
# # plt.ylabel('Y')
# # plt.show()

#====================================
# 其他資料
# YY:農曆年(過年月份) V
# L1: 上個月的市場成交金額 V
# L2: 上上個月的市場成交金額 V
# 市場上的交易日 V