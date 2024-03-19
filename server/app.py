import joblib
import sklearn
import pandas as pd 
import numpy as np
import uvicorn

from fastapi import FastAPI
#from pymongo import MongoClient
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from sklearn.ensemble import RandomForestClassifier

# Create FastAPI instance
app = FastAPI()

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define the structure of the input data
class Item(BaseModel):
    culmen_length: float
    culmen_depth: float
    flipper_length: float
    body_mass: float
    delta_15n: float

@app.get("/")
def root():
    return {"Boop": "Pingoo"}

# Define prediction endpoint
@app.post("/predict")
def predict(item: Item):
    # Convert item to dictionary
    dictionnaire = jsonable_encoder(item)
    
    # Extract features
    X_test = [[dictionnaire[feature] for feature in dictionnaire]]
    #X_test = pd.DataFrame([dictionnaire])
    
    # Make prediction using the loaded model
    prediction = model.predict(X_test)[0]
    
    # Return the predicted class
    return prediction