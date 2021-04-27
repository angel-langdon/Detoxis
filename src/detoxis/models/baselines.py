import os
import pickle
import re

import numpy as np
import pandas as pd
import spacy
from preprocessing import clean_comments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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
        X_aux = self.tfidf.fit_transform(X, y)
        self.clf.fit(X_aux, y)

    def predict(self, X):
        X_aux = self.tfidf.transform(X)
        return self.clf.predict(X_aux)


class SVMClassifier(Baseline):
    def __init__(self):
        super(SVMClassifier, self).__init__()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 5), max_features=20000)
        self.clf = SVC(kernel="linear")

    def fit(self, X, y):
        X_aux = self.tfidf.fit_transform(X, y)
        self.clf.fit(X_aux, y)

    def predict(self, X):
        X_aux = self.tfidf.transform(X)
        return self.clf.predict(X_aux)


class NBClassifier(Baseline):
    def __init__(self):
        super(NBClassifier, self).__init__()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1))
        self.clf = MultinomialNB()

    def fit(self, X, y):
        X_aux = self.tfidf.fit_transform(X, y)
        self.clf.fit(X_aux.todense(), y)

    def predict(self, X):
        X_aux = self.tfidf.transform(X)
        return self.clf.predict(X_aux.todense())
