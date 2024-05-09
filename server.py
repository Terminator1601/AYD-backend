from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained model
with open('./ML model/svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('./ML model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Preprocessing functions
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Endpoint for handling search queries
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    search_query = data['search_query']
    processed_query = preprocess_text(search_query)
    query_vector = tfidf_vectorizer.transform([processed_query])
    predicted_category = model.predict(query_vector)[0]
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
