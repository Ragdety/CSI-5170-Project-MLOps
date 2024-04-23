import logging  
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


class Evaluation(ABC):
    """
    Abstract class for evaluation
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        pass
    

class MSE(Evaluation):
    """
    Evaluation srategy for Mean Squared Error(MSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the MSE

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e

class R2(Evaluation):
    """
    Evaluation srategy for R2
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the R2

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        try:
            logging.info("Calculating R2...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation srategy for Root Mean Squared Error(RMSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Root Mean Squared Error

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        try:
            logging.info("Calculating RMSE...")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE: {e}")
            raise e
    
class Accuracy(Evaluation):
    """
    Evaluation srategy for Accuracy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Accuracy

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        """
        try:
            logging.info("Calculating Accuracy...")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            logging.error(f"Error while calculating Accuracy: {e}")
            raise e
