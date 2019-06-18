# imports
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegressionUsingGD

dataset = pd.read_csv('Exams.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

model = LogisticRegressionUsingGD()
model.fit(X, y, theta)

Y = model.predict(X);
print(Y)

