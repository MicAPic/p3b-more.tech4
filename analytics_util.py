# -*- coding: utf-8 -*-
import re
from collections import Counter
from itertools import chain
from typing import List
from statistics import median
from itertools import product

import gensim.models
import gensim.downloader
import numpy as np
import pandas as pd
import spacy
# import nltk
# import sumy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

POS_TAGS = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
NLP = spacy.load("ru_core_news_sm")


def lemmatize(
        text: str
) -> List[str]:
    """
    Remove links, e-mails and lemmatize the article

    :param text: String containing the full article
    :return: List of lemmas from the given article
    """
    # remove e-mails and url links
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)

    # remove all non-alphabetic characters
    text = re.sub("[^а-яА-Я]+", ' ', text)

    doc = NLP(text)
    text = [token.lemma_ for token in doc if token.pos_ in POS_TAGS]

    return text


def form_ngrams(
        articles: pd.Series
):
    bigram = gensim.models.Phrases(articles)
    trigram = gensim.models.Phrases(bigram[articles])

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    temp = pd.Series(0, index=np.arange(len(articles)))
    for i, entry in articles.items():
        temp[i] = bigram_mod[entry]
        temp[i] = trigram_mod[bigram_mod[entry]]

    return temp


def tf_idf_nitems(
        article: [str],
        n=10
) -> List[str]:
    # get n most important words in the article
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True)
    tf_idf = tf_idf_vectorizer.fit_transform(article)
    results = pd.DataFrame(tf_idf[0].T.todense(), index=tf_idf_vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    results = results.sort_values('TF-IDF', ascending=False)
    return results.head(n).index.to_list()


# used in the following function
lex_rank_summarizer = LexRankSummarizer()
#


def digest(
        text: str
) -> List[str]:
    my_parser = PlaintextParser.from_string(text, Tokenizer('russian'))  # заменить, если будет время
    lexrank_summary = lex_rank_summarizer(my_parser.document, sentences_count=3)
    digest_sentences = []
    for sentence in lexrank_summary:
        digest_sentences.append(str(sentence))
    return digest_sentences


def eval_article(
        w2v_model,
        terms: List[str],
        role_keywords: List[str]
) -> float:
    """
    Evaluate the given article using the similarity with the keywords for a role

    :param w2v_model: Pre-trained W2V model
    :param terms: Most frequent terms of the article for analysis (use TF_IDF)
    :param role_keywords: List of keywords
    :return: Mean value for the article
    """

    results = []

    for article_word, keyword in product(terms, role_keywords):
        try:
            results.append(w2v_model.similarity(article_word + "_" + NLP(article_word)[0].pos_,
                                                keyword + "_" + NLP(keyword)[0].pos_))
        except KeyError:
            results.append(0.0)

    return median(results)


def compare_series(
        old_articles: pd.Series,
        new_articles: pd.Series
) -> (dict, dict):
    """
    Compare two Series of articles (should be already processed by clean_up)

    :param old_articles: 1st series of older data
    :param new_articles: 2nd series of newer data
    :return: Dictionary of trending keywords & dictionary of keywords that are fading away
    """

    old_articles_counted = Counter(list(chain.from_iterable(old_articles)))
    new_articles_counted = Counter(list(chain.from_iterable(new_articles)))

    # get the difference between two dictionaries
    new_articles_counted.subtract(old_articles_counted)
    difference = dict(new_articles_counted.most_common())  # .most_common() sorts the diff dict

    # use k-means to divide the dictionary of keyword popularity difference into 3 clusters
    y_pred = KMeans(n_clusters=3).fit_predict(np.asarray(list(difference.values())).reshape(-1, 1))
    plt.scatter(np.zeros(len(difference)), difference.values(), c=y_pred, marker=".", linewidths=0.1)
    plt.xticks([])
    plt.show()

    # map the two variables to corresponding clusters
    trending_keywords, fading_away_keywords = dict(), dict()
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i - 1]:
            trending_keywords = dict(list(difference.items())[0:i])
            break
    for i in range(len(y_pred) - 2, -1, -1):
        if y_pred[i] != y_pred[i + 1]:
            fading_away_keywords = dict(list(difference.items())[-1:i:-1])
            break

    return trending_keywords, fading_away_keywords


if __name__ == "__main__":
    df = pd.read_csv("temp.tsv", sep="\t")

    # df = df['2022-09-10 00:00:00':"2022-10-10 00:00:00"]
    # sep = df['2022-08-22 00:32:30':"2022-9-9 23:59:59"]

    # sep["Digest"] = sep["Text"].apply(digest)
    # sep["Text"] = sep["Text"].apply(clean_up)
    # sep["Text"] = form_ngrams(sep["Text"])
    # sep["Text"] = sep["Text"].apply(tf_idf_nitems)

    # t, fd = compare_series(sep["Text"], df["Text"])
