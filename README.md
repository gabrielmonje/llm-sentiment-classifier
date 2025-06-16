# Sentiment Classifier with LLM Embeddings

This project demonstrates how to use LLM embeddings to build a simple sentiment analysis model.
It leverages the power of large language models for feature extraction and classic machine learning algorithms for classification.

## Features
- Text embedding using OpenAI's text-embedding-ada-002 model (OpenAI API v1+)
- Sentiment classification (Positive/Negative)
- API with FastAPI
- Web App with Streamlit
- Model training and evaluation

## Tech Stack
- Python
- OpenAI API (v1+)
- Scikit-Learn
- FastAPI
- Streamlit
- Pandas / Numpy
- Matplotlib / Seaborn

## 🔐 API Key Setup
1. Create a `.env` file in the root directory.
2. Add your OpenAI API Key in the following format:
```
OPENAI_API_KEY=your_api_key_here
```
3. This file is ignored by git automatically (.gitignore).

## 🚀 How to Run

### Install dependencies
```
pip install -r requirements.txt
```

### Train the model
```
python src/train_model.py
```

The trained model will be saved in the `model/` folder.

### Run the API (FastAPI)
```
uvicorn api.main:app --reload
```
Access the API at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run the Web App (Streamlit)
```
streamlit run app/streamlit_app.py
```

## License
MIT License
