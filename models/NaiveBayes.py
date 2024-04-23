import numpy as np
from sklearn.base import BaseEstimator

class MyNaiveBayes(BaseEstimator):
  def __init__(self, alpha=1.0):

    """
    Initialize the multinomial naive bayes classifier

    :param alpha: float, default=1.0
    """
    self.alpha = alpha

    # Prior probabilities of each class
    self.class_probs = None

    # Conditional probabilities of each feature given each class
    self.word_probs = None

    # Unique classes
    self.classes = None

  def fit(self, X, y):
    """
    Train the multinomial naive bayes classifier. Returns the trained model.

    :param X: feature matrix of train data
    :param y: list of labels
    :return: self
    """

    self.classes = np.unique(y)
    num_classes = len(self.classes)
    num_words = X.shape[1]

    # Initialize arrays to store the prior and conditional probabilities
    self.class_probs = np.zeros(num_classes)
    self.word_probs = np.zeros((num_classes, num_words))

    for i, cls in enumerate(self.classes):
      num_class_documents = np.sum(y == cls)

      # Calculate the prior probability of each class
      self.class_probs[i] = (num_class_documents + self.alpha) / (len(y) + num_classes * self.alpha)

      word_counts = np.sum(X[y == cls], axis=0)
      total_words_in_class = word_counts.sum()

      # Calculate the conditional probability of each word given each class
      self.word_probs[i] = (word_counts + self.alpha) / (total_words_in_class + num_words * self.alpha)

    return self

  def predict(self, X):
    """
    Predict the class of data in X

    :param X: feature matrix of test data
    :return: class with the highest probability for each sentence
    """
    log_probs = np.zeros((X.shape[0], len(self.classes)))

    for i, cls in enumerate(self.classes):
      # Log probabilities to deal with underflow
      log_probs[:, i] = np.sum(np.log(self.word_probs[i, :]) * X, axis=1) + np.log(self.class_probs[i])

    return np.argmax(log_probs, axis=1)