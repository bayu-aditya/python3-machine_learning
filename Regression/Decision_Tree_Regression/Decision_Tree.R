'
Program     : Decision Tree Regression
Date        : 30 Jul 2018
Author      : Bayu Aditya
'

# Import data
database = read.csv('Position_Salaries.csv')
database = database[2:3]

# Regresi Decision Tree
# install.packages('rpart')
library(rpart)
dec_tree = rpart(formula = Salary ~. , 
                 data = database,
                 control = rpart.control(minsplit = 1))
# summary(dec_tree)
y_pred = predict(dec_tree, newdata = database)

# Plot Data dan hasil regresi
library(ggplot2)
ggplot() + 
  geom_point(aes(x = database$Level, y = database$Salary), color = 'red') +
  geom_line(aes(x = database$Level, y = y_pred), color = 'blue') +
  ggtitle('Decision Tree') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(database$Level), max(database$Level), 0.1)
ggplot() +
  geom_point(aes(x = database$Level, y = database$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(dec_tree, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Decision Tree High Resolution') +
  xlab('Level') +
  ylab('Salary')

# Prediksi Sembarang nilai
y_coba = predict(dec_tree, data.frame(Level = 5.5))
