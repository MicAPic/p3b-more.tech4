'''A little bit of analytical magic'''
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pandas as pd

from analytics_util import (digest, form_ngrams, group_similar, lemmatize,
                            predict_sentiment)


def preprocess_df(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Prepare data for further using

    :param data: data from .tsv file
    :return: data with date, link, title, the num most important words and
    digest columns for each article
    '''
    # remove missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # create digest for an article
    data['Digest'] = data['Text'].map(digest)
    # lemmatization
    tokens, total_sentences, total_syllables = lemmatize(data.Text)
    data['Text_Clean'] = tokens
    data['Text_Clean'] = form_ngrams(data['Text_Clean'])
    data['Text_Clean'] = data.Text_Clean.str.join(' ')

    # evaluation of our metrics
    data['Lexical_Diversity'] = np.fromiter(
        (len(set(tokens)) / len(tokens) for tokens in tokens), float)

    fre = np.fromiter((206.835 - 1.52 * (len(w) / sen) - 65.14 * (syl / len(w))
                       for w, sen, syl in zip(tokens, total_sentences,
                                              total_syllables)), float)
    # 121.22 is the highest possible score;
    data['FRE'] = (fre - np.min(fre)) / (121.22 - np.min(fre))
    # there's no lower bound, so we use the lowest score we got as a reference

    data[['Neutral', 'Positive', 'Negative']] = data.apply(
        lambda row: predict_sentiment(row['Text']), axis=1)

    data['Score'] = data.Lexical_Diversity + data.FRE + \
        data.Neutral + data.Positive - data.Negative

    return data


def eval_data(data: pd.DataFrame, num=3) -> List[List]:
    '''
    Choose the num most popular articles in a dataset. If more than one article
    represents a category, returns the one with the highest overall score.

    :param data: data with articles to choose
    :param n: Number of chosen articles
    :return: Slice of the data w/ n most popular articles
    '''
    groups = group_similar(data['Text_Clean'])
    groups.sort(key=len, reverse=True)

    output = pd.DataFrame(index=range(num), columns=data.columns)
    for i in range(num):
        group = data.iloc[groups[i]]
        group = group.sort_values(by=['Score'], ascending=False)
        output.iloc[i] = group.iloc[0]

    return output.values.tolist()


if __name__ == '__main__':
    dataset = pd.read_csv('presentation/sample_dataset.tsv', sep='\t')
    dataset = preprocess_df(dataset)
    print(eval_data(dataset))
