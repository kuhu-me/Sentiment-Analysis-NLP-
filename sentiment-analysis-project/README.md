Sentiment Analysis on Movie Reviews 🎬🧠

Overview

This project is a Natural Language Processing (NLP) pipeline that performs multi-class sentiment analysis on movie reviews from Rotten Tomatoes. It classifies phrases into five sentiment categories using a machine learning model trained on labeled data.

Sentiment Classes

Each review phrase is labeled with one of the following sentiment values:

0 – Negative
1 – Somewhat Negative
2 – Neutral
3 – Somewhat Positive
4 – Positive

Project Structure

sentiment-analysis-project/
├── data/
│   ├── train.tsv              # Labeled training data
│   ├── test.tsv               # Unlabeled/test data
├── models/
│   ├── model.pkl              # Trained logistic regression model
│   ├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── src/
│   ├── __init__.py            # Exposes prediction interface or initial setup
│   ├── preprocess.py          # Text cleaning, tokenization, lemmatization
│   ├── vectorizer.py          # TF-IDF vectorizer setup and transformation
│   ├── model.py               # Model training and tuning pipeline
│   ├── utils.py               # Optional utility/helper functions
├── requirements.txt           # List of required libraries (e.g., scikit-learn, pandas, nltk)
├── README.md                  # Instructions, setup, and usage info


Features

# Loads and preprocesses text data
# Removes punctuation, stopwords, and lemmatizes tokens
# Converts text into TF-IDF vectors
# Trains a logistic regression model using GridSearchCV
# Saves the trained model and vectorizer
# Predicts sentiment of new text using a clean Python API

"""
Quick Start

    Install Requirements
    pip install -r requirements.txt
    Train the Model
    Run the model training script in src/model.py. It uses GridSearchCV to tune hyperparameters and saves the trained model and vectorizer       to disk.

Predict a Sentiment

    Use the innit file and add the text in the file to predict the sentiment of any new phrase:
    from src/innit import predict_sentiment

print(predict_sentiment("What a beautiful performance!")) # Output: 4 (Positive)

"""


Preprocessing Pipeline

    Text cleaning (punctuation and special characters removed)

    Tokenization

    Stopword removal

    Lemmatization

    Vectorization using TF-IDF (with n-grams)

Modeling Details

    Algorithm: Logistic Regression

    Cross-validation: 5-fold GridSearchCV

    Evaluation metric: Accuracy

    Trained using scikit-learn

Performance

Sample accuracy on the training set: ~74% (depending on hyperparameters and preprocessing)

Dependencies

    pandas

    numpy

    matplotlib

    seaborn

    nltk

    scikit-learn

    joblib

All dependencies are listed in requirements.txt.

*** How to Use This Project ***

   1. As a standalone sentiment prediction module

    Just import predict_sentiment from src/init.py and pass it a phrase.

   2. Extend for web or app integration

    You can wrap the predict_sentiment function into a Flask API or a browser extension for real-world use.

License

    This project is for educational purposes. No licensing for commercial use has been specified.

Acknowledgments

    The Rotten Tomatoes dataset is provided by Stanford University and is widely used for sentiment classification research.
