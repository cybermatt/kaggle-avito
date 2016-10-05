# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import re
import os
import pymorphy2
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import models
from datetime import datetime as dt


def get_similarity_KE(lemmas1, lemmas2):

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

    return ratio


def text_similarity(text1, text2, stemmer, stop_words, model):

    text1 = str(text1)
    text2 = str(text2)

    text1 = text1.replace('\n', '')
    text2 = text2.replace('\n', '')

    lemmas1 = []
    lemmas2 = []

    digits1 = []    # for simbols with digits
    digits2 = []    # for simbols with digits

    # english1 = []   # for brands or english words
    # english2 = []   # for brands or english words

    tokenizer = RegexpTokenizer(r'\w+')
    tk1 = tokenizer.tokenize(text1)
    tk2 = tokenizer.tokenize(text2)

    for word in tk1:
        normal = stemmer.parse(word)[0].normal_form
        # normal = re.search('[а-я]+', normal)

        if not word.isalpha():
            digits1.append(word)
            continue

        # if re.match("^[A-Za-z_-]*$", word):
        #     english1.append(word)
        #     continue

        if word not in stop_words:
            lemmas1.append(normal)

    for word in tk2:
        normal = stemmer.parse(word)[0].normal_form

        if not word.isalpha():
            digits2.append(word)
            continue

        # if re.match("^[A-Za-z_-]*$", word):
        #     english1.append(word)
        #     continue

        if word not in stop_words:
            lemmas2.append(normal)

    try:
        score = model.n_similarity(lemmas1, lemmas2)
    except KeyError as e:
        # print('KEY ERROR', e)
        score = get_similarity_KE(lemmas1, lemmas2)

    # dscore = get_similarity_KE(digits1, digits2)
    # total_score = (score+dscore)/2.0

    return float(score)


def split_train():

    oidx = 1

    for i in range(1, 5):
        fname = 'data/texts/Ptext_' + str(i) + '.csv'

        df = pd.read_csv(fname, compression='gzip')
        splitrow = int(df.shape[0]/2)
        print(df.shape[0])

        set1 = df[:splitrow]
        set1.to_csv('data/texts/splits/train_'+str(oidx)+'.csv', compression='gzip')
        oidx += 1
        print(set1.shape[0])

        set2 = df[splitrow:]
        set2.to_csv('data/texts/splits/train_' + str(oidx) + '.csv', compression='gzip')
        oidx += 1
        print(set2.shape[0])


def split_test():

    oidx = 1

    df = pd.read_csv('data/texts/FPairs_text_test.csv', compression='gzip')
    splitrow = int(df.shape[0]/4)
    print(df.shape[0])

    set1 = df[:splitrow]
    set1.to_csv('data/texts/splits/test_' + str(oidx) + '.csv', compression='gzip')
    oidx += 1
    print(set1.shape[0])

    set2 = df[splitrow:splitrow*2]
    set2.to_csv('data/texts/splits/test_' + str(oidx) + '.csv', compression='gzip')
    oidx += 1
    print(set2.shape[0])

    set3 = df[splitrow*2:splitrow*3]
    set3.to_csv('data/texts/splits/test_' + str(oidx) + '.csv', compression='gzip')
    oidx += 1
    print(set3.shape[0])

    set4 = df[splitrow*3:]
    set4.to_csv('data/texts/splits/test_' + str(oidx) + '.csv', compression='gzip')
    oidx += 1
    print(set4.shape[0])


#
#
#

if __name__ == '__main__':

    set1 = pd.read_csv('data/texts/splits/OUTtrain_1.csv', compression='gzip')
    set2 = pd.read_csv('data/texts/splits/OUTtrain_2.csv', compression='gzip')
    set3 = pd.read_csv('data/texts/splits/OUTtrain_3.csv', compression='gzip')
    set4 = pd.read_csv('data/texts/splits/OUTtrain_4.csv', compression='gzip')
    set5 = pd.read_csv('data/texts/splits/OUTtrain_5.csv', compression='gzip')
    set6 = pd.read_csv('data/texts/splits/OUTtrain_6.csv', compression='gzip')
    set7 = pd.read_csv('data/texts/splits/OUTtrain_7.csv', compression='gzip')
    set8 = pd.read_csv('data/texts/splits/OUTtrain_8.csv', compression='gzip')

    df = pd.concat([set1, set2, set3, set4, set5, set6, set7, set8])
    df.drop([
        'Unnamed: 0', 'Unnamed: 0.1', 'title_1', 'description_1', 'title_2', 'description_2'
    ], axis=1, inplace=True)
    df.to_csv('w2v_train.csv', compression='gzip')

    #
    #
    #

    FILES_PATH = 'data/texts/splits/'
    FILE_ID = '3'
    FILE_NAME = 'train_' + str(FILE_ID) + '.csv'
    SET_IN = os.path.join(FILES_PATH, FILE_NAME)
    SET_OUT = os.path.join(FILES_PATH, 'OUT'+FILE_NAME)

    print('Start...')
    start_time = dt.now().replace(microsecond=0)

    #

    mstem = pymorphy2.MorphAnalyzer()
    stop_words = stopwords.words('russian')
    model = models.Doc2Vec.load_word2vec_format('word2vec/ruscorpora.model.bin', binary=True)
    # model = models.Doc2Vec.load('/mnt/Storage/Word2vec models/all.norm-sz500-w10-cb0-it3-min5.w2v')

    #

    train = pd.read_csv(SET_IN, compression='gzip')

    train['title_simW2'] = train.apply(lambda row: text_similarity(
        row['title_1'], row['title_2'],
        stemmer=mstem,
        stop_words=stop_words,
        model=model
    ), axis=1)

    train['description_simW2'] = train.apply(lambda row: text_similarity(
        row['description_1'], row['description_2'],
        stemmer=mstem,
        stop_words=stop_words,
        model=model
    ), axis=1)

    # print('Title score: {}'.format(roc_auc_score(y_true=train['isDuplicate'], y_score=train['title_simW2'])))
    # print('Descr score: {}'.format(roc_auc_score(y_true=train['isDuplicate'], y_score=train['description_simW2'])))

    # arg1 = 'иду по асфальту я в лыжи обутый'
    # arg2 = 'то ли лыжи не едут то ли я дурак'
    # print(text_similarity(arg1, arg2, mstem, stop_words, model))

    #

    train.to_csv(SET_OUT, compression='gzip')

    end_time = dt.now().replace(microsecond=0)
    print('Elapsed time: {}'.format(end_time - start_time))
    print('Done')


# ['идеальный', 'качество']
# ['кружевной', 'платье']
# ['Новые.', 'Белоснежные.']
# ['Новое,', 'облегающее,']
# -0.0343614607965
# =================================
# ['идеальный', 'качество']
# ['нежный', 'кружево']
# ['Новые.', 'Белоснежные.']
# ['Новое,', 'белоснежное,']
# 0.000826664411575
# =================================
# ['идеальный', 'качество']
# ['бархатный', 'фатина', 'айворя']
# ['Новые.', 'Белоснежные.']
# ['Новое.']
# 0
