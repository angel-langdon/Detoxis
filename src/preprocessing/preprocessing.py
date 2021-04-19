# %%
import os
import re

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.path_utils import paths

# %%


def read_train_dataset():
    train_dataset_path = os.path.join(paths.detoxis, "train.csv")
    df = pd.read_csv(train_dataset_path)
    return df


df = read_train_dataset()

# %%
# Preprocessing


def lower_text(comment):
    return comment.lower()


def delete_stop_words(comment):
    spanish_stopwords = stopwords.words("spanish")
    return " ".join([w for w in comment.split() if w not in spanish_stopwords])


def delete_puntuaction_marks(comment):
    return " ".join(re.findall(r'\w+', comment))


def stemmer(comment):
    stemmer = SnowballStemmer('spanish')

    stemmed_text = [stemmer.stem(i) for i in word_tokenize(comment)]
    return " ".join(stemmed_text)


def separate_underscore_words(comment):
    return " ".join(comment.split("_"))


def separate_dash_words(comment):
    return " ".join(comment.split("-"))


def clean_tweet(comment):
    comment = lower_text(comment)
    comment = delete_stop_words(comment)
    urls = r'https?://[\S]+'
    users = r'@[\S]+'
    hashtags = r'#[\S]+'
    emojis = r'[\U00010000-\U0010ffff]'
    expr = f'''({"|".join([urls,
                           users,
                           hashtags,
                           emojis])})'''
    comment = re.sub(expr, "", comment, flags=re.MULTILINE)  # EMOTICONOS
    comment = delete_puntuaction_marks(comment)
    return stemmer(comment)


def clean_tweet_richer(comment):
    comment = lower_text(comment)
    comment = delete_stop_words(comment)
    comment = separate_dash_words(comment)
    comment = separate_underscore_words(comment)
    urls = r'https?://[\S]+'
    users = r'@[\S]+'
    emojis = r'[\U00010000-\U0010ffff]'
    expr = f'''({"|".join([urls,
                           users,
                           emojis])})'''
    comment = re.sub(expr, "", comment, flags=re.MULTILINE)  # EMOTICONOS
    comment = delete_puntuaction_marks(comment)
    return stemmer(comment)


comment = "El rÁcismo\n #stop_racism #chupa-pollas está mal, no? @w007sya http://www.stopracism.com/index"
clean_tweet(comment)
clean_tweet_richer(comment)


# %%
df["comment_preprocessed"] = [clean_tweet_richer(c) for c in df["comment"]]
# %%
keep_cols = ["thread_id", "comment_id", "reply_to",
             "comment_level", "comment", "comment_preprocessed",
             "toxicity",
             "toxicity_level"]
df = df[keep_cols]
# %%
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(
    df["comment_preprocessed"])
# %%
first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[1]

# place tf-idf values in a pandas data frame
df1 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(),
                   index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df1 = df1.sort_values(by=["tfidf"], ascending=False)
df1.head()
# %%
# Bag of words
bag_vectorizer = CountVectorizer()
bag_vectorizer.fit(df["comment_preprocessed"])
X_bag_of_words = bag_vectorizer.transform(df["comment_preprocessed"])
X_bag_of_words
