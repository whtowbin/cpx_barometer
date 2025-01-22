#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

from lineartree import LinearForestRegressor
from lineartree import LinearBoostRegressor

import numpy as np
#%%




path = "Jorgenson_cpx_dat_MAL_PT_filtered.xlsx"
sheet = 0

df = pd.read_excel(path, sheet_name=sheet)



filtered_col = [ 'P_kbar', 'SiO2_Cpx', 'TiO2_Cpx',
       'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx', 'Na2O_Cpx',
       'Cr2O3_Cpx']




df_filtered = df[filtered_col].dropna()
y_col = df.loc[df_filtered.index]['T_K']


X = df_filtered.to_numpy()
y = y_col
# apply log transfrom to data for normalization
def log_transform(x):
    print(x)
    return np.log(x + 1 ) # np.log(x + 1)


scaler = StandardScaler()
transformer = FunctionTransformer(log_transform)
X = transformer.fit_transform(X)
X = StandardScaler().fit_transform(X)


# Assuming X is your feature data and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=134)




#%%

regr = LinearForestRegressor(base_estimator=LinearRegression(),max_features= 20)
regr.fit(X_train, y_train)
y_test_pred = regr.predict(X_test)

#%%
SEE = np.mean((y_test_pred - y_test.to_numpy())**2)**(1/2)
SEE
# %%
fig, ax = plt.subplots()
ax.plot( y_test, y_test_pred, linestyle = "none", marker = "o", alpha = .4)
ax.set_ylabel("Actual Pressure kbar")
ax.set_ylabel("Predicted Pressure kbar")
# %%
