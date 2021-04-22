# %%
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Preprocessing
def delete_stop_words(comment):
    spanish_stopwords = stopwords.words("spanish")
    return " ".join([w for w in comment.split() if w not in spanish_stopwords])


def steam(text, stemmer):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text)]
    return " ".join(stemmed_text)


def clean_text_column(df, col):
    """Normalizes a string column to have a processed format 
    Arguments:
      df (pd.DataFrame): the dataframe that contains the column to normalize
      col (str): the dataframe column to normalize
      steammer (nltk.steam.SnowballStammer): the steammer to use for 
          steamming the text
    Returns:
      The dataframe with the preprocessed column
    """
    df = df.copy()  #  copy the dataframe avoid modifying the original
    # Initialize the steammer to Spanish language
    stemmer = SnowballStemmer('spanish')
    # Make the comments to lowercase
    df[col] = df[col].str.lower()
    # Delete the stop words
    df[col] = [delete_stop_words(c) for c in df[col]]
    # Replace underscores and hyphens with spaces
    df[col] = df[col].str.replace("_", " ")
    df[col] = df[col].str.replace("-", " ")
    # Create the regex to delete the urls, usernames and emojis
    urls = r'https?://[\S]+'
    users = r'@[\S]+'
    emojis = r'[\U00010000-\U0010ffff]'
    # Join the regex
    expr = f'''({"|".join([urls,
                           users,
                           emojis])})'''
    # Replace the urls, users and emojis with empty string
    df[col] = df[col].str.replace(expr, "", regex=True)
    # Get only the words of the text
    df[col] = df[col].str.findall("\w+").str.join(" ")
    # Delete the numbers
    df[col] = df[col].str.replace("[0-9]+", "", regex=True)
    # Steam the words of the text for each text in the specified column
    df[col] = [steam(c, stemmer) for c in df[col]]
    return df


# read the data
df_original = pd.read_csv("train.csv")
# Normalize the "comment" column
df = clean_text_column(df_original, "comment")
df.head()
# %%
# Bag of words
bag_vectorizer = CountVectorizer()
counts = bag_vectorizer.fit_transform(df["comment"])
counts

# %%
# Change the ngram_range and use chars instead of words
ngram_vectorizer = CountVectorizer(analyzer="char_wb",
                                   ngram_range=(3, 6))
counts = ngram_vectorizer.fit_transform(df["comment"])
counts

# %%
# Create a TF-IDF with ngrams from range 1 to 2
tfidf = TfidfVectorizer(ngram_range=(1, 2))
# Fit with the comments
features = tfidf.fit_transform(df["comment"])
# Get the feature extraction matrix
df_features = pd.DataFrame(features.todense(),
                           columns=tfidf.get_feature_names())
# Print the first comment
print(df_original["comment"].iloc[0])
# Print the sorted by probability first row of the matrix
df_features.sort_values(by=0, axis=1, ascending=False).head(1)

# %%
