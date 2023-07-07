import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")

# Logistic Regression
class LogitRegression06(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.cost_history = []

        for i in range(self.iterations):
            cost = self.compute_cost()
            self.cost_history.append(cost)
            self.update_weights()
        return self

    def compute_cost(self):
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        cost = -(1/self.m) * np.sum(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))
        return cost

    def update_weights(self):
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        Z = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

    def score(self, X, y=None):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
