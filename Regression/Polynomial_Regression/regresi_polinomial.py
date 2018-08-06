"""
Program     : Regresi Polinomial
Date        : 3 Agustus 2018
Author      : Bayu Aditya
"""

import matplotlib.pyplot as plt
import numpy as np

# INPUT DATABASE
import pandas as pd
database = pd.read_csv('Position_Salaries.csv')
level = database.iloc[:, 1:2].values          # sumbu X
gaji = database.iloc[:, 2:].values           # sumbu Y

# Regresi Linear
from sklearn.linear_model import LinearRegression  
lin_reg = LinearRegression().fit(level, gaji)   # class regresi linear
y_pred = lin_reg.predict(level)                 # prediksi terhadap level

# Regresi Polinomial
from sklearn.preprocessing import PolynomialFeatures
x_poly = PolynomialFeatures(degree = 4).fit_transform(level)    # Matriks pangkat
poly_reg = LinearRegression().fit(x_poly, gaji)                 # class regresi
y_pred_poly = poly_reg.predict(x_poly)                          # prediksi terhadap x_poly

# Plot Linear
plt.scatter(level, gaji, color = 'red')
plt.plot(level, y_pred, color = 'blue')
plt.title('Level VS Gaji (Regresi Linear)')
plt.xlabel('Level') ; plt.ylabel('Gaji')
plt.grid(True) ; plt.show()

# Plot Polinomial
plt.scatter(level, gaji, color = 'red')
plt.plot(level, y_pred_poly, color = 'blue')
plt.title('Level VS Gaji (Regresi Polinomial)')
plt.xlabel('Level') ; plt.ylabel('Gaji')
plt.grid(True) ; plt.show()

# Regresi Polinomial HIGH RESOLUTION (n Data)
n = 100
level_high = np.linspace(min(level), max(level), n)             # membuat vektor
level_high = np.reshape(level_high, (len(level_high), 1))       # Reshape ke Matriks
x_poly_high = PolynomialFeatures(degree = 4).fit_transform(level_high)    # Matriks pangkat
y_pred_high = poly_reg.predict(x_poly_high)                     # Prediksi terhadap x_poly_high
# Visualisasi
plt.scatter(level, gaji, color = 'red')
plt.plot(level_high, y_pred_high, color = 'blue')
plt.title('Level VS Gaji (Regresi Polinomial HiGH RESOLUTION)')
plt.xlabel('Level') ; plt.ylabel('Gaji')
plt.grid(True) ; plt.show()
