# -*- coding: utf-8 -*-
import os
import re
from collections import Counter
from datetime import datetime
from itertools import chain, product
from statistics import median
from typing import List

import gensim.downloader
import gensim.models
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from wordcloud import WordCloud
from PIL import Image

POS_TAGS = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
NLP = spacy.load("ru_core_news_lg")


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
        terms: List[str],
        role_keywords: List[str]
) -> float:
    """
    Evaluate the given article using the similarity with the keywords for a role

    :param terms: Most frequent terms of the article for analysis (use TF_IDF)
    :param role_keywords: List of keywords
    :return: Mean value for the article
    """

    results = []
    terms = list(map(NLP, terms))
    role_keywords = list(map(NLP, role_keywords))

    for article_word, keyword in product(terms, role_keywords):
        results.append(article_word.similarity(keyword))

    return median(results)


def find_trends(
        articles: pd.Series,
) -> None:
    """
    Splits the dataframe in half, tries to find trending keywords (should be already processed by clean_up),
    generates a WordCloud image at imgs/word_clouds/

    :param articles: Series of articles from a dataframe
    :return: Dictionary of trending keywords & dictionary of keywords that are fading away
    """

    # divide Series in two
    half_point = int(len(articles / 2))
    old_articles = articles.head(half_point)
    new_articles = articles.tail(half_point - half_point % 2)

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

    # map the variable to corresponding cluster
    trending_keywords = dict()
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i - 1]:
            trending_keywords = dict(list(difference.items())[0:i])
            break

    vtb_mask = np.array(Image.open("imgs/vtb_logo.png"))

    wordcloud = WordCloud(background_color="white",
                          mask=vtb_mask).generate_from_frequencies(frequencies=trending_keywords)
    if not os.path.exists("imgs/word_clouds"):
        os.makedirs("imgs/word_clouds")

    wordcloud.to_file(f"imgs/word_clouds/{datetime.now():%Y-%m-%d-%H%M%S}.jpg")


# if __name__ == "__main__":
#     find_trends(df["Text"])
