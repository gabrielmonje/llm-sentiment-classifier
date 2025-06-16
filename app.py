import streamlit as st
import numpy as np
import joblib
import os
import sys

#  Corrigir path para acessar a pasta src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.embedding_service import get_embedding

#  Carregar modelo treinado
model = joblib.load("model/sentiment_model.joblib")

#  Interface
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

st.title(" Sentiment Classifier with LLM Embeddings")

st.markdown("""
##  Sobre o Projeto

Este app usa **LLM Embeddings da OpenAI + Machine Learning** para classificar se um texto é **positivo** ou **negativo**.

###  Pipeline:
1. O texto é convertido em embeddings (`text-embedding-ada-002` da OpenAI).
2. Um modelo RandomForest classifica o sentimento.
3. O resultado é exibido.

## Como usar:
- Digite um texto em inglês no campo abaixo.
- Clique em **"Analyze Sentiment"**.
- Veja se o sentimento é positivo ou negativo.

---
""")

st.subheader(" Enter a review:")
user_input = st.text_area("Type your review text here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a review text.")
    else:
        with st.spinner("Generating prediction..."):
            embedding = get_embedding(user_input)
            prediction = model.predict(np.array(embedding).reshape(1, -1))
            sentiment = "Positive" if prediction[0] == 1 else "Negative"

            if sentiment == "Positive":
                st.markdown(
                    f"""
                    <div style="background-color:#d4edda;padding:20px;border-radius:10px">
                        <h3 style="color:#155724;"> Predicted Sentiment: {sentiment}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#f8d7da;padding:20px;border-radius:10px">
                        <h3 style="color:#721c24;"> Predicted Sentiment: {sentiment}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
