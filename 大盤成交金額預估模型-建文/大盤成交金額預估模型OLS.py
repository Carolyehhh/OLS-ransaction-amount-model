import statsmodels.api as sm
import numpy as np
import pandas as pd

#duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
#Y = duncan_prestige.data['income']
#X = duncan_prestige.data['education']
#x = sm.add_constant(X)


df=pd.read_excel("成交金額預估.xlsx",sheet_name="model_4")
Y=df["成交億"]
#X=df.iloc[:,2:23]
#X=pd.DataFrame(df,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',"q1","q2","q3","q4",'GDP%',"GDP","人均GDP"])
#X=pd.DataFrame(df,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',
#                          "人均GDP","M1b%","L1","市場",])
X=pd.DataFrame(df,columns=["人均GDP","CPI","M1b%","L1","市場","失業"])

model = sm.OLS(Y,X)
result = model.fit()
print(result.summary())


