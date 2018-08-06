"""
Program     : Random Forest Regression
Date        : 5 Agustus 2018
Author      : Bayu Aditya
"""

# Import Data
import pandas as pd
database = pd.read_csv('Position_Salaries.csv')
level = database.iloc[:, 1:2].values                # Independent Variable (kolom 1)
gaji = database.iloc[:, 2].values                   # Dependent Variable (kolom 2)

# Regresi (Random Forest)
from sklearn.ensemble import RandomForestRegressor
ranfor_reg = RandomForestRegressor(n_estimators = 500)    # Class Random Forest Regression
ranfor_reg.fit(level, gaji)                         # Class berdasarkan 'level'
y_pred = ranfor_reg.predict(level)                  # y prediksi berdasarkan 'level'

# Visualisasi
import matplotlib.pyplot as plt
plt.scatter(level, gaji, color = 'red')
plt.plot(level, y_pred, color = 'blue')
plt.grid(True); plt.title('Random Forest Regression')
plt.xlabel('Levels') ; plt.ylabel('Gaji')
plt.show()

# Regresi Random Forest HIGH RESOLUTION
n = 100
import numpy as np
x_high = np.linspace(min(level), max(level), n)     # sumbu x_high
x_high = np.reshape(x_high, (n, 1))                 # reshape vektor x_high
y_pred_high = ranfor_reg.predict(x_high)            # y prediksi berdasarkan 'x_high'
# Visualisasi HIGH RESOLUTION
plt.scatter(level, gaji, color = 'red')
plt.plot(x_high, y_pred_high, color = 'blue')
plt.grid(True); plt.title('Random Forest Regression (HIGH RESOLUTION)')
plt.xlabel('Levels') ; plt.ylabel('Gaji')
plt.show()