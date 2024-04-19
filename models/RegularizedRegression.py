from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import math
import numpy as np
import warnings

class myRegularizedRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_val=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_val = lambda_val
        self.weights = None
        self.predictions = None
        self.bias = None

    def _sigmoid(self, z):
        warnings.filterwarnings('ignore')
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y, bias):
        num_samples = X.shape[0]
        linear_model = np.dot(X, self.weights) + bias
        self.predictions = self._sigmoid(linear_model)
        loss = -1 / num_samples * (np.dot(y.T, np.log(self.predictions)) + np.dot((1 - y).T, np.log(1 - self.predictions))) \
               + (self.lambda_val / (2 * num_samples)) * np.sum(self.weights ** 2)
        return loss

    def _compute_gradients(self, X, y):
        num_samples = X.shape[0]
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        dw = (1 / num_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_val / num_samples) * self.weights
        db = (1 / num_samples) * np.sum(predictions - y)
        return dw, db

    def train_logistic_regression(self, X_train, y_train):
        num_features = X_train.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            dw, db = self._compute_gradients(X_train, y_train)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights, self.bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return np.round(predictions)


def main():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Shuffle input data
    X, y = shuffle(X, y, random_state=16)

    # training and testing sets split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    my_model = myRegularizedRegression()

    # Train
    best_weights, best_bias = my_model.train_logistic_regression(X_train_scaled, y_train)

    # Evaluate the model

    y_test_pred = my_model.predict(X_test_scaled)
    test_accuracy = np.mean(y_test_pred == y_test)
    test_accuracy = math.ceil(test_accuracy * 100)

    print("Test Accuracy:", test_accuracy, "%")

if __name__=="__main__":
    main()

# def k_fold_cross_validation(X, y, k=5, regularization='l2'):
#     num_samples = X.shape[0]
#     fold_size = num_samples // k
#     accuracies = []
#
#     for i in range(k):
#         # Split data into training and validation sets for this fold
#         val_start = i * fold_size
#         val_end = (i + 1) * fold_size
#         X_val_fold = X[val_start:val_end]
#         y_val_fold = y[val_start:val_end]
#         X_train_fold = np.concatenate([X[:val_start], X[val_end:]])
#         y_train_fold = np.concatenate([y[:val_start], y[val_end:]])
#
#         scaler = StandardScaler()
#         X_train_fold_scaled = scaler.fit_transform(X_train_fold)
#         X_val_fold_scaled = scaler.transform(X_val_fold)
#
#         weights, bias = train_logistic_regression(X_train_fold_scaled, y_train_fold, regularization=regularization)
#
#         y_val_pred = predict(X_val_fold_scaled, weights, bias)
#         accuracy = np.mean(y_val_pred == y_val_fold)
#         accuracies.append(accuracy)
#
#     avg_accuracy = np.mean(accuracies)
#     return avg_accuracy


# lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]  # regulariztion coefficient
#
# # grid search 6 cross
# best_lambda = None
# best_accuracy = 0
#
# for lambda_val in lambda_values:
#     accuracy = k_fold_cross_validation(X_train_scaled, y_train, k=5, regularization='l2')
#     print(f"Regularization coefficient lambda: {lambda_val}, Cross-validation accuracy: {accuracy}")
#
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_lambda = lambda_val
#
# print(f"Best regularization coefficient lambda: {best_lambda}")


