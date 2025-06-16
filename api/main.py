from fastapi import FastAPI
from pydantic import BaseModel
from src.embedding_service import get_embedding
import numpy as np
import joblib
import os

app = FastAPI()

# Load model
model = joblib.load("model/sentiment_model.joblib")


class TextInput(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "LLM Sentiment Classifier API is running"}


@app.post("/predict")
def predict_sentiment(input: TextInput):
    embedding = get_embedding(input.text)
    prediction = model.predict(np.array(embedding).reshape(1, -1))
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return {"sentiment": sentiment}
