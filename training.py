import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lemmatizer = WordNetLemmatizer()

# 1. Cargar los datos del archivo JSON
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: El archivo 'intents.json' no fue encontrado.")
    exit()

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# 2. Preprocesar los datos
for intent in intents['intenciones']:
    for pattern in intent['patrones']:
        # Tokenizar y lematizar
        word_list = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters]
        
        # Unir las palabras lematizadas en una sola cadena
        processed_pattern = ' '.join(lemmatized_words)
        
        words.extend(lemmatized_words)
        documents.append((processed_pattern, intent['etiqueta']))
        
        if intent['etiqueta'] not in classes:
            classes.append(intent['etiqueta'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Guardar las clases para usarlas después
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# 3. Crear los datos de entrenamiento
training_patterns = [doc[0] for doc in documents]
training_tags = [doc[1] for doc in documents]

# Vectorizar los patrones de texto
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_patterns)
y_train = np.array(training_tags)

# Guardar el vectorizador
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# 4. Entrenar el modelo de clasificación
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("¡Entrenamiento completado! Los archivos del modelo han sido creados.")