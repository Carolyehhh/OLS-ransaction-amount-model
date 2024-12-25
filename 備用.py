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
# # OLS Model 預測
# X_train, X_test, Y_train, Y_test = train_test_split(X_lag, Y_lag, test_size=0.2, random_state=42)
# # 添加截距項
# X_train = sm.add_constant(X_train)

# # 訓練 OLS 模型
# model_lag = sm.OLS(Y_train, X_train)
# reset_lag = model_lag.fit()

# 查看訓練結果
# print(reset_lag.summary())

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