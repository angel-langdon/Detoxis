# %%
import re

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def clean_comments(X):
    def delete_stop_words(comment):
        spanish_stopwords = stopwords.words("spanish")
        return " ".join([w for w in comment.split() if w not in spanish_stopwords])

    def steam(text, stemmer):
        stemmed_text = [stemmer.stem(word) for word in word_tokenize(text)]
        return " ".join(stemmed_text)

    stemmer = SnowballStemmer('spanish')

    def clean(string):
        string = string.lower()
        string = delete_stop_words(string)
        string = string.replace("_", " ").replace("-", " ")
        urls = r'https?://[\S]+'
        users = r'@[\S]+'
        emojis = r'[\U00010000-\U0010ffff]'
        hashtags = r'\s#[\S]+'
        # Join the regex
        expr = f'''({"|".join([urls,
                               users,
                               emojis,
                               hashtags])})'''
        string = re.sub(expr, " ", string)
        string = " ".join(re.findall("\w+", string))
        string = re.sub("[0-9]+", "", string)
        string = steam(string, stemmer)
        return string

    return np.array([clean(c) for c in X])
