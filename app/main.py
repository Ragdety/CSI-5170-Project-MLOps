from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import CountVectorizer
from zenml_loader import *

# app = FastAPI()

def load_naive_model():
  return load_artifact_by_version('80093327-33ba-410e-bd52-5eb48ef99a44')

def load_logistic_model():
  return load_artifact_by_version('3ef73e90-2c3f-4433-a9b4-1dacda7a71a6')

def load_vectorizer():
  return load_artifact_by_version('bdc1121d-d922-4b33-80f1-92d09ed17bee')

# @app.post("/prediction")
# def predict(request: dict):
#   # Load the model
#   model = load_model()
  
#   # Make a prediction for Naive Bayes model
#   prediction = model.predict(request)
  
#   return JSONResponse(content={"prediction": prediction})


if __name__ == "__main__":
  # Load the model
  # model = load_naive_model()
  model = load_logistic_model()

  print("Model: ", model)

  vectorizer = load_vectorizer()
  email = ["Congratulations! You've won a prize. Claim it now."]
  email = ["U can call me now"]
  print("Email: ", email)

  new_email = vectorizer.transform(email)
  
  # Make a prediction for Naive Bayes model
  prediction = model.predict(new_email)
  print("Prediction: ", prediction)


  if prediction[0] == 0:
    print("This email is not spam")
  else:
    print("This email is spam")