import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Data Import
from Market_Transaction_OLS_Prediction import data

mkt_amount = data[2]['成交金額(千)_月']
mkt_amount = pd.DataFrame(mkt_amount)

# 1. 檢查數據是否平穩（差分）
# 我們可以對市場成交金額進行差分處理，確保數據是平穩的
# 如果時間序列中有趨勢或季節性，使用差分來移除

# 檢查時間序列的平穩性，若數值平穩則不需進行微分
# plot_acf(mkt_amount) # 判斷為非平穩序列：這種緩慢衰減的 ACF 特徵表明該時間序列可能具有「趨勢」或「非平穩性」。需要進行差分 .diff() 來使序列變得平穩。
# plot_pacf(mkt_amount) # 
# plt.show()

plot_acf(mkt_amount.diff())
plot_pacf(mkt_amount.diff())
plt.show()