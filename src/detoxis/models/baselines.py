import os
import pickle
import re

import numpy as np
import pandas as pd
import spacy
from preprocessing import clean_comments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.evaluation import Evaluator
from utils.inout import write_results


class Baseline:
    def __init__(self):
        self.evaluator = Evaluator()

    def predict(self, X):
        return np.ones((X.shape[0],))

    def score(self, X, y, save=False):
        y_pred = self.predict(X)
        if save:
            write_results(y, 'gold_standard.txt')
            write_results(y_pred, 'predictions.txt')
        return self.evaluator.evaluate(y, y_pred)


class RandomClassifier(Baseline):
    def __init__(self):
        super(RandomClassifier, self).__init__()

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(0, 4, (X.shape[0],))


class BOWClassifier(Baseline):
    def __init__(self):
        super(BOWClassifier, self).__init__()
        self.tfidf = TfidfVectorizer(
            strip_accents='unicode', lowercase=True, ngram_range=(1, 2), max_features=5000)
        self.classifier = SVC(kernel='linear')

    def fit(self, X, y):
        X_aux = self.tfidf.fit_transform(X, y)
        self.classifier.fit(X_aux, y)

    def predict(self, X):
        X_aux = self.tfidf.transform(X)
        return self.classifier.predict(X_aux)


class LRClassifier(Baseline):
    def __init__(self):
        super(LRClassifier, self).__init__()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1))
        self.clf = LogisticRegression(C=12, max_iter=1000)

    def fit(self, X, y):
        X = clean_comments(X)
        X_aux = self.tfidf.fit_transform(X, y)
        self.clf.fit(X_aux, y)

    def predict(self, X):
        X = clean_comments(X)
        X_aux = self.tfidf.transform(X)
        return self.clf.predict(X_aux)


class SVMClassifier(Baseline):
    def __init__(self):
        super(SVMClassifier, self).__init__()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1))
        self.clf = SVC(gamma="auto")

    def fit(self, X, y):
        X = clean_comments(X)
        X_aux = self.tfidf.fit_transform(X, y)
        self.clf.fit(X_aux, y)

    def predict(self, X):
        X = clean_comments(X)
        X_aux = self.tfidf.transform(X)
        return self.clf.predict(X_aux)


class ChainBOW(Baseline):
    def __init__(self):
        super(ChainBOW, self).__init__()
        self.bow_toxic = BOWClassifier()
        self.bow_degree = BOWClassifier()

    def fit(self, X, y):
        # The first bow model needs to detect which comments are toxic
        self.bow_toxic.fit(X, (y > 0).astype(int))

        # The second bow model takes the toxic detected comments and has to determine the toxicity degree
        self.bow_degree.fit(X[y > 0], y[y > 0])

    def predict(self, X):
        out = self.bow_toxic.predict(X)
        toxic_cond = out > 0
        out[toxic_cond] = self.bow_degree.predict(X[toxic_cond])
        return out


class Word2VecSpacy(Baseline):
    def __init__(self):
        super(Word2VecSpacy, self).__init__()
        self.nlp = spacy.load("es_core_news_lg")
        self.classifier = SVC(kernel='linear')

    @staticmethod
    def _sent2vec(sent):
        return sent.vector

    def fit(self, X, y):
        X_vecs = np.array([self._sent2vec(doc) for doc in self.nlp.pipe(X)])
        self.classifier.fit(X_vecs, y)

    def predict(self, X):
        X_vecs = np.array([self._sent2vec(doc) for doc in self.nlp.pipe(X)])
        return self.classifier.predict(X_vecs)


class GloVeSBWC(Baseline):
    def __init__(self):
        super(GloVeSBWC, self).__init__()
        glove_file = './data/embeddings-l-model.vec'
        if not os.path.isfile(glove_file):
            glove_file_aux = './data/glove-sbwc.i25.vec'
            with open(glove_file_aux, 'r', encoding='utf-8') as f:
                self.n_tokens, self.vec_dim = next(f)[:-1].split(' ')
                self.glove_vecs = pd.read_csv(
                    f, sep=" ", index_col=0, header=None)
                self.glove_vecs = self.glove_vecs.T.to_dict('list')
            with open(glove_file, 'wb') as f:
                pickle.dump(self.glove_vecs, f)
        else:
            with open(glove_file, 'rb') as f:
                self.glove_vecs = pickle.load(f)
            self.n_tokens, self.vec_dim = len(
                self.glove_vecs), len(self.glove_vecs['de'])
        self.nlp = spacy.load("es_core_news_lg")
        self.classifier = SVC(kernel='linear')

    def _sent2vec(self, sent):
        vec = np.zeros((self.vec_dim,))
        for tok in sent:
            vec += self.glove_vecs.get(tok.text.lower(),
                                       np.zeros((self.vec_dim,)))
        return vec / len(sent)

    def fit(self, X, y):
        X_vecs = np.array([self._sent2vec(doc) for doc in self.nlp.pipe(X)])
        self.classifier.fit(X_vecs, y)

    def predict(self, X):
        X_vecs = np.array([self._sent2vec(doc) for doc in self.nlp.pipe(X)])
        return self.classifier.predict(X_vecs)
