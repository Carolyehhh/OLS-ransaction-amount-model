# Goal:大盤成交金額預估，目標是用前一期的數值預估下一期的「市場成交金額」
# 資料從201701-202410(因為202411的部分資料，如:CPI、M1B尚未公布)、相比於建文的檔案排除WPI相關數值，因為「主計總處自資料時間112年1月起停編躉售物價指數」
# WPI 改使用 PPI >> 使用「中華民國統計資訊網」的資料
# 2023年前使用 WPI，2023年後使用 PPI
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
lagged_columns = ['是否放假', 'log_貨幣總額年增率(M1B)期底', 'log_貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率', 
                  '上個月成交金額(千)_月', '上上個月成交金額(千)_月', 'log_交易天數_月','經濟成長率(GDP)–單季', 'log_生產者物價指數']
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

# OLS Model 預測
X_train, X_test, Y_train, Y_test = train_test_split(X_lag, Y_lag, test_size=0.2, random_state=42)
# 添加截距項
X_train = sm.add_constant(X_train)

# 訓練 OLS 模型
model_lag = sm.OLS(Y_train, X_train)
reset_lag = model_lag.fit()

# 查看訓練結果
# print(reset_lag.summary())


""""
TO DO:
預測有error，整個滯後預測的過程待釐清。
"""
# 在測試集上進行預測
X_test = sm.add_constant(X_test)  # 確保包含截距
Y_pred = reset_lag.predict(X_test)
# 檢查預測結果
# print(Y_pred) #50    9.807510e+09 可能異常

#使用滯後數據進行未來預測
X_future = Combined_lag.loc[Combined_lag['年月'] == '202411'].copy()  # 202411 是最後一個月
X_future['上個月成交金額(千)_月'] = Combined_lag.loc[Combined_lag['年月'] == '202410', '成交金額(千)_月'].values
X_future['上上個月成交金額(千)_月'] = Combined_lag.loc[Combined_lag['年月'] == '202409', '成交金額(千)_月'].values
#進行未來數據預測
X_future = sm.add_constant(X_future)  # 添加截距
Y_future_pred = reset_lag.predict(X_future)

print(X_train.shape)  # 應該輸出 (樣本數量, 11)
print(X_future.shape)

# print("Predicted future transaction amount:", Y_future_pred)


#---------------------------------------------------------------------------------------------------------------------
# 部分變數的VIF仍高( log_平均每人國內生產毛額(GDP)–美元、log_貨幣總額年增率(M2)期底、中央銀行重貼現率等)，進行正規化 => 找尋有幫助的變數
# "Ridge" 正規化 (使用X3的變數)
# Y3 = Combined_df['成交金額(千)_月']

# # 標準化數據
# scaler = StandardScaler()
# X3_scaled = scaler.fit_transform(X3)

# # 拆分訓練集和測試集
# X3_train, X3_test, y3_train, y3_test = train_test_split(X3_scaled, Y3, test_size=0.2, random_state=42)

# # 訓練Ridge回歸模型
# ridge3 = Ridge(alpha=1.0) # alpha是正則化強度，越大正則化越強
# ridge3.fit(X3_train, y3_train)
# y_pred = ridge3.predict(X3_test)

# # print(y3_test.shape)
# # print(y_pred)

# # 評估模型
# # print("Ridge模型訓練集得分(R^2):", ridge3.score(X3_train, y3_train)) # 0.8845，模型能夠解釋約 88.45% 的訓練數據中目標變數的變異性。
# # print("Ridge模型測試集得分(R^2):", ridge3.score(X3_test, y3_test)) # 0.7675，模型在測試數據上也有不錯的表現，但比訓練數據稍低。
# # print(X3.columns)
# # print(ridge3.coef_)

# # 標準化數據
# # 在回到 OLS 模型之前，將原始數據進行標準化是必要的，特別是當 Ridge 使用了標準化數據時
# scaler = StandardScaler()
# X3_train_scaled = scaler.fit_transform(X3_train)
# X3_test_scaled = scaler.transform(X3_test)
# =====================================
# # 解決殘差非常態分佈的問題: 
# # 偏度計算
# skewness_values = Combined_df.select_dtypes(include='number').apply(lambda x: stats.skew(x, bias=False))
# # print(skewness_values)
# # 針對偏度大的變數優先處理(>1、<-1)
# # 右偏，取對數處理
# high_skewness = skewness_values[skewness_values > 1]
# for col in high_skewness.index:
#     Combined_df[col + '_log'] = np.log1p(Combined_df[col]) # Log(1+x) 避免 Log(0)
# # print("High", high_skewness)
# # 左偏，平方根處理
# low_skewness = skewness_values[skewness_values < -1]
# for col in low_skewness.index:
#     Combined_df[col + '_sqrt'] = np.sqrt(Combined_df[col])
# # print("Low", low_skewness)

# # 取對數&平方根處理後，不偏性沒有明顯的改善
# filter_columns = Combined_df.filter(like='_sqrt').columns.tolist() + Combined_df.filter(like='_log').columns.tolist()
# skewness_values2 = Combined_df[filter_columns].apply(lambda x: stats.skew(x, bias=False))
# # print("transform", skewness_values2)

# # 針對偏性仍很大的變數進行 Box-Cox
# Combined_df['交易天數_月_boxcox'], _ = boxcox(Combined_df['交易天數_月'])
# Combined_df['失業率_boxcox'], _ = boxcox(Combined_df['失業率'])

# # delete useless columns
# Combined_df = Combined_df.drop(['交易天數_月_sqrt', '失業率_log'], axis=1)
# # print(Combined_df.columns)

# # # 計算變換後的偏度
# # skewness_v = stats.skew(Combined_df['交易天數_月_boxcox'], bias=False) # -0.2446
# # skewness_vv = stats.skew(Combined_df['失業率_boxcox'], bias=False) # -0.0685
# # print(f"Box-Cox後的偏度: {skewness_vv}")

# # 使用解決過"偏性"的變數做回歸
# X1 = Combined_df[['是否放假_log', '經濟成長率(GDP)–單季', '國內生產毛額(GDP)–美元',
#        '平均每人國內生產毛額(GDP)–美元', '消費者物價指數(CPI)',
#        '消費者物價指數(CPI)年增率', '貨幣總額(M1B)日平均', '貨幣總額年增率(M1B)期底_log', '貨幣總額(M2)日平均',
#        '貨幣總額年增率(M2)期底', '中央銀行重貼現率', '失業率_boxcox', '交易天數_月_boxcox', '上個月成交金額(千)_月',
#        '上上個月成交金額(千)_月']]

# # 添加截距項，當自變數都為0時，Y的期望值 => 提升模型表現力(更好地擬合)、正確解釋模型參數、兼容性(符合標準的回歸公式)
# X1 = sm.add_constant(X1)

# Y1 = Combined_df['成交金額(千)_月']

# # OLS Model
# transaction_amount_model1 = sm.OLS(Y1, X1)
# result1 = transaction_amount_model1.fit()
# print(result1.summary())
# # 結論：基於我的目的是"預測"市場金額，殘差的"偏性"可能對於預測結果影響不大，加上使用處理過的變數導致殘差變得更非常態分佈，故先不處理這部分


# # 對Y(市場成交金額)預估
# Y_pred_train = result0.predict(X)
# # print(Y_pred_train)

# 比較:預測值 vs 實際值
# actual_vs_predicted = pd.DataFrame({
#     "Actual":Y,
#     "Predicted":Y_pred_train,
#     "Difference": Y_pred_train - Y
# })
# print(len(actual_vs_predicted)) #94

# 殘差項
# A.常態分佈
# sns.histplot(residuals, kde=True) # kde=True 繪製核密度曲線
# plt.show()
# B.變異數同質性
# C.各自獨立



# =================
# 建文原使的 code
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

#====================================
# # 檢查資料的分布形狀
# sns.histplot(Combined_df['失業率'], kde=True)
# plt.title('Histogram of unemployee rate')
# plt.show()