# -*- coding: utf-8 -*-
from typing import List

import pandas as pd

from analytics_util import lemmatize, remove_similar, form_ngrams, \
    tf_idf_nitems, digest, eval_article

# keywords that form a description of the role's job
ROLE_KEYWORDS = {
    'Accountant': ['бухгалтер', 'закон', 'налог', 'счет-фактура',
                   'инвойс', 'бухгалтерия', 'смета'],
    'CEO': ['директор', 'закон', 'предприниматель', 'предпринимательство',
            'бизнес', 'бизнесмен', 'дотация', 'налог']
}


def preprocess_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataframe for further usage

    :param dataframe: Dataframe from .tsv file
    :return: Dataframe with date, link, title, the n most important words and digest columns for each article
    """

    # remove missing values and duplicates
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(inplace=True)
    # set "Date" as the index and sort
    dataframe.set_index('Date', inplace=True)
    dataframe.sort_index(inplace=True)

    # create digest for an article
    dataframe['Digest'] = dataframe['Text'].map(digest)
    # find n most important words of an article
    dataframe['Text'] = dataframe['Text'].map(lemmatize)
    dataframe['Text'] = form_ngrams(dataframe['Text'])
    dataframe['Text'] = dataframe['Text'].map(tf_idf_nitems)
    dataframe = remove_similar(dataframe)

    return dataframe


def eval_data_4_role(role: str, dataframe: pd.DataFrame, n=3) -> List[List]:
    """
    Picks n most relevant articles for a role by its keywords

    :param role: String containing the name of the role
    :param dataframe: Dataframe with articles to evaluate
    :param n: No. of chosen articles
    :return: n most relevant articles for a role
    """
    assert ROLE_KEYWORDS[role]
    dataframe[role] = dataframe['Text'].map(
        lambda x: eval_article(terms=x, role_keywords=ROLE_KEYWORDS[role]))
    dataframe = dataframe.sort_values(role, ascending=False).head(n=n)

    return dataframe.drop(['Title', 'Text', role], axis=1).values.tolist()


if __name__ == '__main__':
    df = pd.read_csv('temp.tsv', sep='\t')
    df = preprocess_df(df)
    res = eval_data_4_role('CEO', df)
    # pass
