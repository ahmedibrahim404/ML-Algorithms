import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LinearRegression:


    def computeCost(self, X, Y, theta):
        m = len(Y)
        predictions = X.dot(theta)
        square_err = (predictions - Y) ** 2

        return 1 / (2 * m) * np.sum(square_err)


    def gradientDescent(self, X, Y, theta, alpha, num_iters):

        m = len(Y)
        J_history = []

        for i in range(num_iters):
            predictions = X.dot(theta)
            error = np.dot(X.transpose(), (predictions - Y))
            descent = alpha * 1 / m * error
            theta -= descent
            J_history.append(self.computeCost(X, Y, theta))

        return theta, J_history



    def linear_regression(self, X, Y):
        m = len(X)
        X = np.append(np.ones((m, 1)), X.reshape(m, 1), axis=1)
        Y = Y.reshape(m, 1)
        theta = np.zeros((2, 1))
        theta, J_history = self.gradientDescent(X, Y, theta, 0.01, 1500)
        xVals = [x for x in range(25)]
        yVals = [y * theta[1] + theta[0] for y in xVals]
        return xVals, yVals

