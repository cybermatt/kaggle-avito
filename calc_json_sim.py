# -*- coding: utf-8 -*-
__author__ = 'Matt Stroganov'

import json
import pandas as pd
import numpy as np

from datetime import datetime as dt
from sklearn.metrics import roc_auc_score


def get_similarity_json(inp1, inp2, dup=0):

    if type(inp1) == float:
        return 0

    json1 = json.loads(inp1)
    json2 = json.loads(inp2)

    if len(json1) > len(json2):
        big = json1
        small = json2
    else:
        big = json2
        small = json1

    big_len = len(big)

    # empty
    if big_len == 0:
        return 0

    kidx = 0
    vidx = 0

    for key in big.keys():
        if key in small.keys():
            # kidx += 1
            if big[key] == small[key]:
                vidx += 1

    # TODO: without kidx
    # ratio = np.exp((((kidx + vidx)/big_len) / 2.0) + 0.5)
    # ratio = ((kidx + vidx)/big_len) / 2.0
    # ratio = (len(inters)/len(lemmas1) + len(inters)/len(lemmas2)) / 2.0

    ratio = (vidx / big_len) / 2.0

    return ratio


def json_sim_set(set_in, set_out):

    train = pd.read_csv(set_in, compression='gzip')

    train['json_sim'] = train.apply(lambda row: get_similarity_json(row['attrsJSON_x'], row['attrsJSON_y'], 0), axis=1)

    train.drop([
        'title_1', 'description_1', 'attrsJSON_x',
        'title_2', 'description_2', 'attrsJSON_y',
    ], axis=1, inplace=True)

    # print('AUC score: {}'.format(roc_auc_score(y_true=train['isDuplicate'], y_score=train['json_sim'])))

    train.to_csv(set_out, compression='gzip')


if __name__ == "__main__":
    print('Start...')
    start_time = dt.now().replace(microsecond=0)
    #

    # js1 = '{"Марка":"ВАЗ (LADA)", "Модель":"2110", "Тип автомобиля":"С пробегом", "Год выпуска":"2002", "Пробег":"50 000 - 54 999", "Тип кузова":"Седан", "Цвет":"Серебряный", "Объём двигателя":"1.6", "Коробка передач":"Механика", "Количество дверей":"5", "Тип двигателя":"Бензин", "Привод":"Передний", "Руль":"Левый", "Состояние":"Не битый", "Мощность двигателя":"92"}'
    # js2 = '{"Марка":"ВАЗ (LADA)", "Модель":"2110", "Тип автомобиля":"С пробегом", "Год выпуска":"2002", "Пробег":"110 000 - 119 999", "Тип кузова":"Седан", "Цвет":"Серебряный", "Объём двигателя":"1.6", "Коробка передач":"Механика", "Количество дверей":"4", "Тип двигателя":"Бензин", "Привод":"Передний", "Руль":"Левый", "Состояние":"Не битый", "Мощность двигателя":"94"}'

    # js1 = '{"Вид телефона": "MTS", "Марка":"ВАЗ (LADA)", "Модель":"2110"}'
    # js2 = '{"Вид телефона": "Другие марки"}'

    # js1 = '{"Тип объявления":"Сдам", "Количество комнат":"1", "Срок аренды":"На длительный срок", "Комиссия":"Нет", "Залог":"0", "Этаж":"2", "Этажей в доме":"9", "Тип дома":"Панельный", "Площадь":"35", "Адрес":"черняховского 32"}'
    # js2 = '{"Тип объявления":"Сдам", "Количество комнат":"1", "Срок аренды":"На длительный срок", "Комиссия":"Нет", "Залог":"0", "Этаж":"2", "Этажей в доме":"9", "Тип дома":"Панельный", "Площадь":"35", "Адрес":"Черняховского ул, 32", "Кол-во кроватей":"1"}'

    # get_similarity_json(js1, js2)

    train_in = 'data/texts/Ptext_train.csv'
    train_out = 'data/texts/FPairs_JSON_onlvidx_train.csv'
    # train_out = 'data/texts/FPairs_JSON_kidxvidx_train.csv'
    json_sim_set(train_in, train_out)

    test_in = 'data/texts/FPairs_text_test.csv'
    test_out = 'data/texts/FPairs_JSON_onlvidx_test.csv'
    # test_out = 'data/texts/FPairs_JSON_kidxvidx_test.csv'
    json_sim_set(test_in, test_out)

    #
    end_time = dt.now().replace(microsecond=0)
    print('Elapsed time: {}'.format(end_time-start_time))
    print('Done')
