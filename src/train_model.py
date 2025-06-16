import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from embedding_service import get_embedding
from data_processing import load_data, preprocess_text

# Criar a pasta 'model' se não existir
os.makedirs('model', exist_ok=True)

#  Carregar os dados
df = load_data("data/imdb_sample.csv")
df = preprocess_text(df, "review")

#  Gerar os embeddings
print("Generating embeddings...")
df['embedding'] = df['review'].apply(lambda x: get_embedding(x))

#  Preparar dados para treino
X = np.vstack(df['embedding'].values)
y = df['sentiment'].map({'positive': 1, 'negative': 0})

#  Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

#  Avaliar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#  Salvar o modelo na pasta 'model'
joblib.dump(clf, 'model/sentiment_model.joblib')
print(" Modelo salvo em: model/sentiment_model.joblib")
