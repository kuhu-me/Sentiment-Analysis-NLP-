''' IMPORTING ALL THE LIBRARIES'''
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess

tfidf=TfidfVectorizer(ngram_range=(1, 2))

matrix = tfidf.fit_transform(preprocess.data_set["Lemmitized_phrase"])
print(f""" TF-IDF matrix: (train)\n\n{matrix}\n\n""")
matrix_test = tfidf.fit_transform(preprocess.data_frm["Lemmitized_phrase"])
print(f""" TF-IDF matrix: (test)\n\n{matrix_test}\n\n""")


# DATA Loading for training in model

Y_train = preprocess.data_set['Sentiment']
X_train = matrix
X_test = matrix_test
print(f"""train and test:\n\n{X_train,X_test}\n""")