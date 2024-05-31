from app import app
import nltk


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    app.run(debug=True ,port=3000)
