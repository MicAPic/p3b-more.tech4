import copy
import os
from datetime import datetime
from typing import List

import pandas as pd
from gensim.models import Word2Vec


def train_model(
        articles: pd.Series
):
    """
    Train a Word2Vec model based on a dataframe of articles (should be already cleaned up) + Lenta.Ru dataset.

    The model is saved at /models/

    :param articles: Path to the file
    """
    new_model = Word2Vec(sentences=articles, vector_size=300, min_count=1, workers=4, sg=1)
    if not os.path.exists("models"):
        os.makedirs("models")
    new_model.save(f"models/w2v_{datetime.now():%Y-%m-%d-%H%M%S}.model")


def find_most_similar(
        w2v_model,
        target_word: str,
        n=7
) -> List[str]:
    """
    Search for matches that are similar in meaning to the given target_word among the keywords of a pre-trained
    W2V model (see https://en.wikipedia.org/wiki/Word2vec)

    :param w2v_model: Pre-trained W2V model
    :param target_word: Word for search
    :param n: No. of matches to return
    :return: List of n best matches
    """
    if target_word not in w2v_model.index_to_key:
        w2v_model = copy.deepcopy(w2v_model)
        w2v_model.build_vocab([[target_word]], update=True)

    target_word = target_word.replace(' ', '_')
    corpus = w2v_model.key_to_index.keys()
    similarity_values = w2v_model.most_similar(target_word, topn=None)
    res = dict(zip(corpus, similarity_values))
    # print first n most similar keywords to the target_word
    return sorted(res, key=res.__getitem__, reverse=True)[1:n]
