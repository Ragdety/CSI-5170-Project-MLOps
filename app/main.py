from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import CountVectorizer
from zenml_loader import *
from pydantic import BaseModel


class PredictionRequest(BaseModel):
  email: str

class PredictionResponse(BaseModel):
  is_spam: bool
  prediction: int
  message: str


def load_naive_model():
  return load_artifact_by_version('80093327-33ba-410e-bd52-5eb48ef99a44')

def load_logistic_model():
  return load_artifact_by_version('3ef73e90-2c3f-4433-a9b4-1dacda7a71a6')

def load_svm_model():
  return load_artifact_by_version('f29fc0ae-080f-4111-a453-c0f700cfdd66')

def load_vectorizer():
  return load_artifact_by_version('bdc1121d-d922-4b33-80f1-92d09ed17bee')


app = FastAPI()


@app.post("/prediction")
def predict(request: PredictionRequest):
  # Load the model
  model = load_naive_model()
  # model = load_logistic_model()

  print("Model: ", model)

  vectorizer = load_vectorizer()
  email = [request.email]
  email = vectorizer.transform(email)
  print("Email: ", request.email)
  
  # Make a prediction for Naive Bayes model
  prediction = model.predict(email)
  pred = prediction[0]
  is_spam = pred == 1
  message = ""

  print("Prediction: ", pred)

  if is_spam:
    message = "This email is spam"
    print(message)
  else:
    message = "This email is not spam"
    print(message)
  
  return PredictionResponse(is_spam=is_spam, prediction=pred, message=message)
