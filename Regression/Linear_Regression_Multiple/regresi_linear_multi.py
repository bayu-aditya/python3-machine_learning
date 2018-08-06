"""
Program     : Multiple Linear Regression
Tanggal     : Mon Jul 23 14:58:42 2018
Author      : Bayu Aditya
"""

import pandas as pd
import numpy as np

# Menginput database dokumen csv
database = pd.read_csv('50_Startups.csv')
x = database.iloc[:, 0:4].values            # Input kolom 0 - 3 (independent)
y = database.iloc[:, 4:].values             # Input kolom 4 (dependent)

# Encode nama negara (kolom 3)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])               # Encode kolom 3
onehotencoder = OneHotEncoder(categorical_features = [3])   # Dummy Variabel
x = onehotencoder.fit_transform(x).toarray()                # apply ke kolom 3

# Mengurangi variabel encoder jadi N - 1
x = x[:, 1:]

# Split data train dan set (executable) 
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, test_size = 0.2, random_state = 0)
y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 0)

# Regresi Linear Multiple
from sklearn.linear_model import LinearRegression
regresilinear = LinearRegression()
regresilinear.fit(x_train, y_train)

# Data hasil regresi linear
y_pred = regresilinear.predict(x_train)


import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
# Tahap 1 ------------------------------------------
hasil = sm.OLS(endog = y, exog = x_opt).fit()
hasil.summary()
# Tahap 2 ------------------------------------------
x_opt = x[:, [0, 1, 3, 4, 5]]
hasil = sm.OLS(endog = y, exog = x_opt).fit()
hasil.summary()
# Tahap 3 ------------------------------------------
x_opt = x[:, [0, 3, 4, 5]]
hasil = sm.OLS(endog = y, exog = x_opt).fit()
hasil.summary()
# Tahap 4 ------------------------------------------
x_opt = x[:, [0,3,5]]
hasil = sm.OLS(endog = y, exog = x_opt).fit()
hasil.summary()
# Tahap 5 ------------------------------------------
x_opt = x[:, [0,3]]
hasil = sm.OLS(endog = y, exog = x_opt).fit()
hasil.summary()
