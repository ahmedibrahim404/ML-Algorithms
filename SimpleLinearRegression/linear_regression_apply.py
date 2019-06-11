
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

ln = LinearRegression()
x_value, y_value=  ln.linear_regression(X, Y)

plt.plot(x_value,y_value,color="r")
plt.scatter(X, Y)
plt.xlabel("Experience ")
plt.ylabel("Salary")
plt.title("Experience Vs Salary")
plt.show()