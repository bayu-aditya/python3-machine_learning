"""
Program     : Decision Tree Regression
Date        : 5 Agustus 2018
Author      : Bayu Aditya
"""

# Import Data
import pandas as pd
database = pd.read_csv('Position_Salaries.csv')
level = database.iloc[:, 1:2].values                    # independent variable
gaji = database.iloc[:, 2:].values                      # dependent variable

# Regresi Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor().fit(level, gaji)     # Class Regresi Tree
y_pred = tree_reg.predict(level)                        # Regresi berdasarkan 'level'

# Visualisasi
import matplotlib.pyplot as plt
plt.scatter(level, gaji, color = 'red')
plt.plot(level, y_pred, color = 'blue')
plt.title('Decision Tree') ; plt.grid(True)
plt.xlabel('Level') ; plt.ylabel('Gaji')
plt.show()

# Regresi Decision Tree (HIGH RESOLUTION) n data
n = 100                                                 # n data
import numpy as np 
x_high = np.linspace(min(level), max(level), n)         # x_high
x_high = np.reshape(x_high, (n,1))                      # reshape matriks x_high
y_pred_high = tree_reg.predict(x_high)                  # Regresi berdasarkan 'x_high'

# Visualisasi (HIGH RESOLUTION)
plt.scatter(level, gaji, color = 'red')
plt.plot(x_high, y_pred_high, color = 'blue')
plt.title('Decision Tree (HIGH RESOLUTION)') ; plt.grid(True)
plt.xlabel('Level') ; plt.ylabel('Gaji')
plt.show()