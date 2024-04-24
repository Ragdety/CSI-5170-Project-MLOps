from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from abc import ABC, abstractmethod
from models.LogisticRegression import MyLogisticRegression
from models.NaiveBayes import MyNaiveBayes


class Model(ABC):
    @abstractmethod 
    def train(self, X_train, y_train):
        pass


class MyLogisticRegressionModel(Model):
    def train(self, X_train, y_train):
        model = MyLogisticRegression()
        trained_model = model.fit(X_train, y_train)
        return trained_model
    
class LogisticRegressionModel(Model):
    def train(self, X_train, y_train):
        model = LogisticRegression()
        trained_model = model.fit(X_train, y_train)
        return trained_model

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        model = LinearRegression()
        trained_model = model.fit(X_train, y_train)
        return trained_model

class MyNaiveBayesModel(Model):
    def train(self, X_train, y_train):
        model = MyNaiveBayes()
        trained_model = model.fit(X_train, y_train)
        return trained_model
    
class NaiveBayesModel(Model):
    def train(self, X_train, y_train):
        model = MultinomialNB()
        trained_model = model.fit(X_train, y_train)
        return trained_model
    
class SVMModel(Model):
    def train(self, X_train, y_train):
        model = SVC(kernel = 'rbf')
        trained_model = model.fit(X_train, y_train)
        return trained_model