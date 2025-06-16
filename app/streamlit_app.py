import streamlit as st
import numpy as np
import joblib
import os
import sys

#  Adicionar o diretório raiz no sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding_service import get_embedding

#  Carregar o modelo treinado
model = joblib.load("model/sentiment_model.joblib")

#  Interface
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

st.title(" Sentiment Classifier with LLM Embeddings")

st.markdown("""
##  Sobre o Projeto

Este aplicativo usa **LLM Embeddings da OpenAI + Machine Learning** para classificar se o sentimento de um texto é **positivo** ou **negativo**.

1. O texto é convertido em embeddings pela OpenAI (`text-embedding-ada-002`).
2. Um modelo de RandomForest classifica o sentimento.
3. O resultado é exibido aqui.

##  Como usar:
- Digite um texto em inglês no campo abaixo.
- Clique em **"Analyze Sentiment"**.
- Veja o resultado abaixo.

---
""")

st.subheader(" Enter a review:")
user_input = st.text_area("Type your review text here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning(" Please enter a review text.")
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
