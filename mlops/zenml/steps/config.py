from zenml.steps import BaseParameters 
from constants import *

class ModelNameConfig(BaseParameters):
  model_name : str = SVM