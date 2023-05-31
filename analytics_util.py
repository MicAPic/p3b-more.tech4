'''A little more analytical magic'''
# -*- coding: utf-8 -*-

import copy
from itertools import chain, combinations, product
from statistics import median
from typing import List

import gensim.models
import numpy as np
import pandas as pd
import spacy
import torch
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

POS_TAGS = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
VOWELS = frozenset('аоиыуэ')
NLP = spacy.load('ru_core_news_lg')
LRS = LexRankSummarizer()

TOKENIZER = BertTokenizerFast.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment')
MODEL = AutoModelForSequenceClassification.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment')

THRESHOLD = 0.915


def predict_sentiment(text) -> pd.Series:
    inputs = TOKENIZER(text, max_length=512, padding=True,
                       truncation=True, return_tensors='pt')
    outputs = MODEL(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)

    return pd.Series(predicted.detach().numpy().flatten())


def count_vowels(text) -> int:
    return sum(letter in VOWELS for letter in text)


def lemmatize(articles: pd.DataFrame) -> tuple:
    # remove e-mails and url links
    articles = articles.replace(r'\S*@\S*\s?', '', regex=True)
    articles = articles.replace(r'http\S+', '', regex=True)
    articles = articles.replace(r'www.\S+', '', regex=True)

    # fix punctuation
    articles = articles.replace(r'\.([а-яА-Я])', r'. \g<1>', regex=True)

    # remove all non-alphabetic characters
    articles = articles.replace(r'[^а-яА-Я.]+', ' ', regex=True)

    # lowercase
    articles = articles.str.lower()

    docs = list(NLP.pipe(articles, batch_size=len(articles)))

    tokens = [[token.lemma_ for token in article if token.pos_ in POS_TAGS]
              for article in docs]
    sentence_count = [len(list(article.sents)) for article in docs]
    # in russian, no. of vowels ~= no. of syllables
    syllable_count = [count_vowels(article) for article in articles]

    return tokens, sentence_count, syllable_count


# def lemmatize(article: str) -> List[str]:
#     '''
#     Removes links, e-mails and lemmatize the article.
#     Used in eval_data_4_role()
#
#     :param article: String containing the full article
#     :return: List of tokenized articles
#     '''
#     # remove e-mails and url links
#     article = re.sub(r'\S*@\S*\s?', '', article)
#     article = re.sub(r'http\S+', '', article)
#     article = re.sub(r'www.\S+', '', article)
#
#     # remove all non-alphabetic characters
#     article = re.sub('[^а-яА-Я]+', ' ', article)
#
#     doc = NLP(article)
#     article = [token.lemma_ for token in doc if token.pos_ in POS_TAGS]
#
#     return article


def remove_similar(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    If the dataframe contains articles that are very similar in meaning,
    removes the one with the longer digest.
    Unused.

    :param dataframe: Full dataframe received from the bot
    :return The same dataframe, but now without implicitly similar articles
    '''
    temp_series = copy.deepcopy(dataframe['Digest'])
    temp_series = temp_series.map(lambda x: NLP(' '.join(x)))

    digest_pairs = list(combinations(temp_series, r=2))
    for digest1, digest2 in digest_pairs:
        if digest1.similarity(digest2) >= 0.87:
            # keep the article with the shorter digest; nobody has
            # the time to read these days
            candidate = digest1 if len(digest1) > len(digest2) else digest2
            # dataframe = dataframe[dataframe['Digest'] != candidate]
            try:
                dataframe = dataframe.drop(
                    temp_series.index[temp_series == candidate])
            except KeyError:
                continue

    return dataframe


def group_similar(clean_text: pd.Series) -> List[List[int]]:
    '''
    If the dataframe contains articles that are very similar in meaning,
    group them into one category.
    Used in eval_data().

    :param clean_text: Lemmatized articles
    :return Indices of the articles grouped into categories
    '''
    docs = list(NLP.pipe(clean_text, batch_size=len(clean_text)))

    def get_similarity(index1, index2) -> float:
        return docs[index1].similarity(docs[index2])

    res = []
    added_articles = set()

    for i in range(len(clean_text)):
        for j in range(i + 1, len(clean_text)):
            if get_similarity(i, j) > THRESHOLD:
                added_articles.add(i)
                added_articles.add(j)

                if i and j in chain(*res):
                    continue

                new_group = {i, j}
                for group in res:
                    if i in group:
                        flag = True
                        for element in group:
                            if get_similarity(element, j) < THRESHOLD:
                                new_group.remove(i)
                                flag = False
                        if flag:
                            group.append(j)
                            new_group.clear()
                            break
                    elif j in group:
                        flag = True
                        for element in group:
                            if get_similarity(element, i) < THRESHOLD:
                                new_group.remove(j)
                                flag = False
                        if flag:
                            group.append(i)
                            new_group.clear()
                            break

                if new_group:
                    res.append(list(new_group))

    for i in list(clean_text.index.values):
        if i not in added_articles:
            res.append([i])

    return res


def form_ngrams(articles: pd.Series) -> pd.Series:
    '''
    Substitutes collocations with respective n-grams where it can be applied.
    Used in eval_data_4_role()

    :param articles: Series containing the necessary tokenized articles
    :return: Series with n-grams added
    '''
    bigram = gensim.models.Phrases(articles)
    trigram = gensim.models.Phrases(bigram[articles])

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    temp = pd.Series(0, index=np.arange(len(articles)))
    for i, entry in articles.items():
        temp[i] = bigram_mod[entry]
        temp[i] = trigram_mod[bigram_mod[entry]]

    return temp


def digest(article: str, num=3) -> List[str]:
    '''
    Forms a digest based on the article given. Used in eval_data_4_role()

    :param article: String containing the full article
    :param n: No. of sentences to return
    :return: List of num sentences that summarize the article
    '''
    my_parser = PlaintextParser.from_string(article, Tokenizer('russian'))
    lexrank_summary = LRS(my_parser.document, sentences_count=num)
    digest_sentences = []

    for sentence in lexrank_summary:
        digest_sentences.append(str(sentence))

    return digest_sentences


def eval_article(terms: List[str], role_keywords: List[str]) -> float:
    '''
    Evaluate the given article based on the similarity with the keywords
    for a role. Used in eval_data_4_role()

    :param terms: Most frequent terms of the article for analysis (use TF-IDF)
    :param role_keywords: List of keywords
    :return: Mean value for the article
    '''
    results = []
    terms = list(map(NLP, terms))
    role_keywords = list(map(NLP, role_keywords))

    for article_word, keyword in product(terms, role_keywords):
        results.append(article_word.similarity(keyword))

    return median(results)


if __name__ == '__main__':
    pass
