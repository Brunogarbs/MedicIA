import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy
import joblib

# Carregar o dataset CSV
df = pd.read_csv('dataset_expansao.csv')

# Carregar o modelo de idioma do SpaCy para português
nlp = spacy.load("pt_core_news_sm")

# Separar os dados em recursos (X) e rótulos (y)
X = df['sintomas']
y = df['doenca']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obter stopwords em português e convertê-las para lista
stop_words = list(nlp.Defaults.stop_words)

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treinamento do modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train_tfidf, y_train)

# Previsões no conjunto de teste
y_pred = modelo.predict(X_test_tfidf)

# Avaliar a acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# Salvar o modelo e o vetor TF-IDF
joblib.dump(modelo, 'modelo_sintomas.pkl')
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')
