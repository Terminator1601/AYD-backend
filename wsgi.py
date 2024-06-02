from app import app
import nltk

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    app.run(host='0.0.0.0', port=8000)
