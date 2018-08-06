"""
Program     : Membuat regresi Linear
Tanggal     : Sat Jul 21 09:11:44 2018
Author      : Bayu Aditya
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import Data
database = pd.read_csv('Salary_Data.csv')
x = database.iloc[:, :-1].values                # Pengalaman bekerja (years)
y = database.iloc[:, 1].values                  # Gaji

# Pemecahan data train dan test (execute)
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, test_size = 1/3, random_state = 0)
y_train, y_test = train_test_split(y, test_size = 1/3, random_state = 0)

# Proses Regresi Linear
from sklearn.linear_model import LinearRegression
regresi_linear = LinearRegression()
regresi_linear.fit(x_train, y_train)

# Data untuk garis regresi linear (dari data train)
y_regresi = regresi_linear.predict(x_train)

# Plot data train dan garis regresi linear
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, y_regresi, color = 'blue')
plt.title('Regresi Linear (Train)'); plt.xlabel('Pengalaman (tahun)'); plt.ylabel('Gaji')
plt.grid(True)
plt.show()

# Plot data test (execute) dan garis regresi linear
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, y_regresi, color = 'blue')
plt.title('Regresi Linear (Data)'); plt.xlabel('Pengalaman (tahun)'); plt.ylabel('Gaji')
plt.grid(True)
plt.show()
