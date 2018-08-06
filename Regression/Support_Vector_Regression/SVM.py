"""
Program     : Support Vector Regression (SVR)
Date        : 4 Agustus 2018
Author      : Bayu Aditya
"""
import numpy as np

# Import Data
import pandas as pd
database = pd.read_csv('Position_Salaries.csv')
level = database.iloc[:, 1:2].values                # Level (independent var)
salary = database.iloc[:, 2:].values                # Salary (dependent var)

# Normalisasi
from sklearn.preprocessing import StandardScaler
x_class = StandardScaler()                          # Class normalisasi x
y_class = StandardScaler()                          # Class normalisasi y
x = x_class.fit_transform(level)
y = y_class.fit_transform(salary)

# Regresi SVR
from sklearn.svm import SVR
svr_reg = SVR().fit(x,y)                            # Class SVR
y_pred = svr_reg.predict(x)                         # Regresi berdasarkan x
y_pred = y_class.inverse_transform(y_pred)          # Roll back dari Normalisasi

# Visualisasi Data
import matplotlib.pyplot as plt
plt.scatter(level, salary, color = 'red')
plt.plot(level, y_pred, color = 'blue')
plt.grid(True) ; plt.title('Support Vector Regression')
plt.xlabel('Levels') ; plt.ylabel('Salary')
plt.show()

# ______________REGRESI DATA (HIGH RESOLUTION n data)________________________
n = 100
x_high = np.linspace(min(level), max(level), n)     # Membuat sumbu X_high
x_high = np.reshape(x_high, (n,1))                  # Reshape vaktor X_high

# Normalisasi (HIGH Resolution)
x_high_class = StandardScaler()                     # Class normalisasi x_high
x_high = x_high_class.fit_transform(x_high)         # Normalisasi x_high
y_high = svr_reg.predict(x_high)                    # Regresi berdasarkan X_high
y_high = y_class.inverse_transform(y_high)          # Roll back normalisasi dari class y sebelumnya
x_high = x_high_class.inverse_transform(x_high)     # Roll back normalisasi dari class x_high
# Visualisasi Data (HIGH RESOLUTION n data)
plt.scatter(level, salary, color = 'red')
plt.plot(x_high, y_high, color = 'blue')
plt.grid(True) ; plt.title('Support Vector Regression (HIGH RESOLUTION)')
plt.xlabel('Levels') ; plt.ylabel('Salary')
plt.show()