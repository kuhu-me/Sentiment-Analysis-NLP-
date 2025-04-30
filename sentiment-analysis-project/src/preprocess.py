''' IMPORTING ALL THE LIBRARIES'''

import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



data_set=pd.read_table('C:/Users/KUHU/sentiment_analysis_main/sentiment-analysis-project/data/train.tsv')
data_frm=pd.read_table('C:/Users/KUHU/sentiment_analysis_main/sentiment-analysis-project/data/test.tsv')

# Download nltk
# nltk.download('punkt_tab')
# Download stopwords
# nltk.download('stopwords')
# Download wordnet
# nltk.download('wordnet')


lem=WordNetLemmatizer()


# Null value count

print(f"""train data null value count:\n\n{data_set.isnull().sum()}\n\n""")

print(f"""train data null value count:\n\n{data_frm.isnull().sum()}\n\n""")


# Additional column "Phrase Length"

data_set["Phrase_Length"] = data_set["Phrase"].apply(lambda x: len(x.split()))
print(f"""Phrase length train:\n\n{data_set["Phrase_Length"].head(2)}\n\n""")
print(f"""Phrase length train description:\n\n{data_set["Phrase_Length"].describe()}\n\n""")

data_frm["Phrase_Length"] = data_frm["Phrase"].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
print(f"""Phrase length test:\n\n{data_frm["Phrase_Length"].head(2)}\n\n""")
print(f"""Phrase length test description:\n\n{data_frm["Phrase_Length"].describe()}\n\n""")


# checks for the existence of alphabets or chars in phrase



alphabets = data_set["Phrase"].str.contains(r"[A-Za-z]", regex=True, na=False)
chars=data_set['Phrase'].str.contains(r"[^\w\s]", regex=True, na=False)

print(f"""alphabets in train:\n\n{data_set[~alphabets]}\n\n""")
print(f"""chars in train:\n\n{data_set[chars]}\n\n""")

alphabets = data_frm["Phrase"].str.contains(r"[A-Za-z]", regex=True, na=False)
chars_test=data_frm['Phrase'].str.contains(r"[^\w\s]", regex=True, na=False)

print(f"""alphabets in test:\n\n{data_frm[~alphabets]}\n\n""")
print(f"""chars in test:\n\n{data_frm[chars]}\n\n""")



# checks if emojis are there in the phrases


def emojis():
    emoji_pattern = r"["
    emoji_pattern += "\U0001F600-\U0001F64F"  # Emoticons
    emoji_pattern += "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    emoji_pattern += "\U0001F680-\U0001F6FF"  # Transport & map symbols
    emoji_pattern += "\U0001F1E0-\U0001F1FF"  # Flags
    emoji_pattern += "]"

    emoji_rows = data_set[data_set["Phrase"].str.contains(emoji_pattern, regex=True)]

    print(f"""emoji existence in train:\n\n{emoji_rows}\n\n""")

emojis()

def emojis():
    emoji_pattern = r"["
    emoji_pattern += "\U0001F600-\U0001F64F"  # Emoticons
    emoji_pattern += "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    emoji_pattern += "\U0001F680-\U0001F6FF"  # Transport & map symbols
    emoji_pattern += "\U0001F1E0-\U0001F1FF"  # Flags
    emoji_pattern += "]"

    emoji_rows = data_frm[data_frm["Phrase"].isna() | data_frm["Phrase"].str.contains(emoji_pattern, regex=True)]

    print(f"""emoji existence in test:\n\n{emoji_rows}\n\n""")

emojis()



# converts phrase to lower case

data_set["Phrase"] = data_set["Phrase"].str.lower()
data_frm["Phrase"] = data_frm["Phrase"].str.lower()


# remove punctuations

data_set["Phrase"] = data_set["Phrase"].apply(lambda x: re.sub(r"[^\w\s]", "", x) if isinstance(x, str) else ""
)
data_frm["Phrase"] = data_frm["Phrase"].apply(lambda x: re.sub(r"[^\w\s]", "", x) if isinstance(x, str) else ""
)

# add new column containing tokens of phrase

data_set["Tokens"] = data_set["Phrase"].apply(lambda x:word_tokenize(x))
data_frm["Tokens"] = data_frm["Phrase"].apply(lambda x:word_tokenize(x) if pd.notnull(x) else [] )

stop_words=set(nltk.corpus.stopwords.words('english'))
print(f"""stop words default:\n\n{stop_words}\n\n""")

# remove stop words from corpus

data_set['Tokens']=data_set['Tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
data_frm['Tokens']=data_frm['Tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])


# Lemmitization

data_set["Lemmitized_Tokens"] = data_set["Tokens"].apply(lambda tokens: [lem.lemmatize(word) for word in tokens])
data_frm["Lemmitized_Tokens"] = data_frm["Tokens"].apply(lambda tokens: [lem.lemmatize(word) for word in tokens])
data_set["Lemmitized_phrase"] = data_set["Lemmitized_Tokens"].apply(lambda tokens: " ".join(tokens))
data_frm["Lemmitized_phrase"] = data_frm["Lemmitized_Tokens"].apply(lambda tokens: " ".join(tokens))
