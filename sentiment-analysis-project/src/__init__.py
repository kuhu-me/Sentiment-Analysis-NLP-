from model import best_model
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import stop_words,word_tokenize,lem


# Download nltk
# nltk.download('punkt_tab')
# Download stopwords
# nltk.download('stopwords')
# Download wordnet
# nltk.download('wordnet')


def preprocess_text(text):
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word.lower() not in stop_words]
    tokens=[lem.lemmatize(word) for word in tokens]
    tokens=" ".join(tokens)

    return tokens


def predict_sentiment(text: str) -> int:
    """ Given a raw text string, returns the predicted sentiment label (0–4). """
    cleaned = preprocess_text(text)
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    features = tfidf.transform([cleaned])
    label = best_model.predict(features)[0]

    return label

all = ["predict_sentiment"]