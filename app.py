

from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('./ML model/ML_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('./ML model/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Define the category mapping
category_mapping = {0: 'Hotel', 1: 'Restaurant', 2: 'Gym', 3: 'Coaching', 4: 'Spa', 5: 'Consultant'}

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '')

    # Preprocess the query
    processed_query = preprocess_text(query)

    # Transform the query using the TF-IDF vectorizer
    query_vector = tfidf_vectorizer.transform([processed_query])

    # Predict the category
    predicted_category = model.predict(query_vector)[0]
    predicted_category_name = category_mapping[predicted_category]

    # Return the prediction as a JSON response
    return jsonify({'category': predicted_category_name})



@app.route("/api/healthchecker", methods=["GET"])
def healthchecker():
    return {"status": "success", "message": "Integrate Flask Framework with Next.js"}

