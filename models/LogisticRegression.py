import numpy as np
from sklearn.base import BaseEstimator
import warnings

class MyLogisticRegression(BaseEstimator):
  def __init__(self, learning_rate=0.01, n_iter=1000, lambda_val=0.1):
    self.learning_rate = learning_rate
    self.n_iter = n_iter
    self.lambda_val = lambda_val

    self.weights = None
    self.bias = 0

  def _sigmoid(sefl, z):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-z))
  
  def _compute_loss(self, X, y):
    num_samples = X.shape[0]
    linear_model = np.dot(X, self.weights) + self.bias
    predictions = self._sigmoid(linear_model)
    loss = -1/num_samples * (np.dot(y.T, np.log(predictions)) + np.dot((1 - y).T, np.log(1 - predictions))) \
           + (self.lambda_val / (2 * num_samples)) * np.sum(self.weights ** 2)
    return loss

  def _compute_gradients(self, X, y):
      num_samples = X.shape[0]
      linear_model = np.dot(X, self.weights) + self.bias
      predictions = self._sigmoid(linear_model)

      dw = (1/num_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_val / num_samples) * self.weights
      db = (1/num_samples) * np.sum(predictions - y)

      return dw, db

  def fit(self, X, y):
      num_features = X.shape[1]
      self.weights = np.zeros(num_features)

      for _ in range(self.n_iter):
          dw, db = self._compute_gradients(X, y)
          self.weights -= self.learning_rate * dw
          self.bias -= self.learning_rate * db

      return self

  def predict(self, X):
      linear_model = np.dot(X, self.weights) + self.bias
      predictions = self._sigmoid(linear_model)
      return np.round(predictions)