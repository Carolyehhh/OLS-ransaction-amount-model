# 變數篩選，適用於變數較少時，便於理解模型選擇的過程
import statsmodels.api as sm
import pandas as pd
from Market_Transaction_OLS_Prediction import Combined_lag


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

X = Combined_lag[feature_columns]
Y = Combined_lag['成交金額(千)_月']

# 加入常數項
X = sm.add_constant(X)

# 使用後向刪除逐步迴歸
def backward_elimination(X, Y, significance_level=0.05):
    while True:
        model = sm.OLS(Y, X).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            exclude_feature = p_values.idxmax()
            X = X.drop(columns=[exclude_feature])
            print(f"移除變數: {exclude_feature}, p-value: {max_p_value}")
        else:
            break
    return model

final_model = backward_elimination(X, Y)
print(final_model.summary())