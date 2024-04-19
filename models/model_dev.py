import logging 
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from models.RegularizedRegression import myRegularizedRegression


class Model(ABC):
    @abstractmethod 
    def train(self, X_train, y_train):
        pass


class RegularizedRegressionModel(Model):
    def train(self, X_train, y_train):
        my_model = myRegularizedRegression()
        trained_RR = my_model.train_logistic_regression(X_train, y_train)
        return trained_RR

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        their_model = LinearRegression()
        trained_LR = their_model.fit(X_train, y_train)
        return trained_LR
