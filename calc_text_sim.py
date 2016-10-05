# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import re
import pymorphy2
import pandas as pd

from rutermextract import TermExtractor
from fuzzywuzzy import fuzz
from pymystem3 import Mystem
from sklearn.metrics import roc_auc_score
from datetime import datetime as dt


def get_similarity(arg1, arg2):

    arg1 = str(arg1)
    arg2 = str(arg2)

    arg1 = arg1.replace('\n', '')
    arg2 = arg2.replace('\n', '')

    term_extractor = TermExtractor()

    subterms1 = term_extractor(arg1, nested=True)
    subterms2 = term_extractor(arg2, nested=True)

    ratio = 0

    average_length = (len(subterms1) + len(subterms2)) / 2
    if average_length == 0:
        return 0

    set1 = set(subterms1)
    set2 = set(subterms2)

    intersection = set.intersection(set1, set2)

    ratio += len(intersection)

    set1_ = set.symmetric_difference(set1, intersection)
    set2_ = set.symmetric_difference(set2, intersection)

    # for term1 in set1_:
    #     for term2 in set2_:
    #         # rat = fuzz.ratio(term1.normalized, term2.normalized)
    #         rat = fuzz.partial_ratio(term1.normalized, term2.normalized)
    #         if rat > 30:
    #             ratio += rat*0.01

    metric = ratio/average_length   # TODO: mean or smth else
    # metric = np.mean(ratio)

    return metric


def get_similarity_1(mstem, arg1, arg2, dup=1):

    # TODO: penalty for short ads

    arg1 = str(arg1)
    arg2 = str(arg2)

    arg1 = arg1.replace("\n", ' ')
    arg2 = arg2.replace("\n", ' ')

    lemmas1 = []
    for word in arg1.split(' '):
        lemmas1.append(mstem.parse(word)[0].normal_form)

    lemmas2 = []
    for word in arg2.split(' '):
        lemmas2.append(mstem.parse(word)[0].normal_form)

    # lemmas1 = mstem.lemmatize(arg1)
    # lemmas2 = mstem.lemmatize(arg2)

    big = lemmas2
    small = lemmas1
    if len(lemmas1) > len(lemmas2):
        big = lemmas1
        small = lemmas2

    inters = [i for i in small if i in big]     # TODO: or conversely

    # no intersection
    if len(inters) == 0:
        return 0

    ratio = (len(inters)/len(lemmas1) + len(inters)/len(lemmas2)) / 2.0

    # avg_len = (len(lemmas1) + len(lemmas2)) / 2.0
    # ratio -= (1.0/avg_len)  # penalty

    return ratio


def text_sim_set(set_in, set_out, train=True):

    train = pd.read_csv(set_in, compression='gzip')

    # mstem = Mystem()
    mstem = pymorphy2.MorphAnalyzer()

    train['title_sim'] = train.apply(
        lambda row: get_similarity_1(mstem, row['title_1'], row['title_2'], 0), axis=1
    )
    train['description_sim'] = train.apply(
        lambda row: get_similarity_1(mstem, row['description_1'], row['description_2'], 0), axis=1
    )

    train.drop([
        'title_1', 'description_1', 'attrsJSON_x',
        'title_2', 'description_2', 'attrsJSON_y',
    ], axis=1, inplace=True)

    if train:
        # print(roc_auc_score(y_true=train['isDuplicate'], y_score=train['title_sim']))
        print('Title score: {}'.format(roc_auc_score(y_true=train['isDuplicate'], y_score=train['title_sim'])))
        print('Descr score: {}'.format(roc_auc_score(y_true=train['isDuplicate'], y_score=train['description_sim'])))

    train.to_csv(set_out, compression='gzip')


if __name__ == "__main__":
    print('Start...')
    start_time = dt.now().replace(microsecond=0)
    #

    train_in = 'data/texts/FPairs_train.csv.gzip'
    train_out = 'data/texts/FPairs_text_train.csv.gzip'
    text_sim_set(train_in, train_out)

    test_in = 'data/texts/FPairs_test.csv.gzip'
    test_out = 'data/texts/FPairs_text_test.csv.gzip'
    text_sim_set(test_in, test_out, train=False)

    #
    end_time = dt.now().replace(microsecond=0)
    print('Elapsed time: {}'.format(end_time-start_time))
    print('Done')
