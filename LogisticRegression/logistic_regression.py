import numpy as np
from scipy.optimize import fmin_tnc


class LogisticRegressionUsingGD:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        return np.dot(x, theta)

    def probability(self, theta, x):
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, theta):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=(x, y.flatten()))
        self.w_ = opt_weights[0]
        return self

    def predict(self, x):
        theta = self.w_[:, np.newaxis]
        return self.probability(theta, x)
