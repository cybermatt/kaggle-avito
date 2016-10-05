# -*- coding: utf-8 -*-
import datetime
import random
import time
import pickle
import zipfile
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from operator import itemgetter
from sklearn.metrics import roc_auc_score

random.seed(707)

INPUT_DIR = 'data/'

dhash = pickle.load(open('data/hashes/dhash.pkl', 'rb'))
whash = pickle.load(open('data/hashes/whash.pkl', 'rb'))


def hDist(s1, s2):
    s1, s2 = str(s1), str(s2)
    hDist = sum(bool(ord(ch1) - ord(ch2)) for ch1, ch2 in zip(s1, s2))
    return hDist


def flowdist(a1, a2):
    a1 = [int(x) for x in str(a1).split(',') if x.strip().isdigit()]
    a2 = [int(x) for x in str(a2).split(',') if x.strip().isdigit()]
    flowdist = 99
    for ar1 in a1:
        if ar1 in dhash:
            for ar2 in a2:
                if ar2 in dhash:
                    z = hDist(dhash[ar1], dhash[ar2])
                    if z < flowdist:
                        flowdist = int(z)
    return flowdist


def flowdist2(a1, a2):
    a1 = [int(x) for x in str(a1).split(',') if x.strip().isdigit()]
    a2 = [int(x) for x in str(a2).split(',') if x.strip().isdigit()]
    flowdist_ = 99
    for ar1 in a1:
        if ar1 in whash:
            for ar2 in a2:
                if ar2 in whash:
                    z = hDist(whash[ar1], whash[ar2])
                    if z < flowdist_:
                        flowdist = int(z)
    return flowdist


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def run_default_test(train, test, features, target, random_state=0):

    eta = 0.1
    max_depth = 10
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }

    num_boost_round = 333
    early_stopping_rounds = 5
    test_size = 0.2

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)

    y_train = X_train[target]
    y_valid = X_valid[target]

    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [
        (dtrain, 'train'),
        (dvalid, 'eval')
    ]

    gbm = xgb.train(
        params, dtrain,
        num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=True
    )

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_ntree_limit)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(
        xgb.DMatrix(test[features]),
        ntree_limit=gbm.best_ntree_limit
    )

    print('Training time: {} minutes'.format(round((time.time() - start_time) / 60, 2)))

    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submissions/xgb_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('id,probability\n')
    total = 0
    for id in test['id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):

    trainval = list(train.columns.values)
    testval = list(test.columns.values)

    output = intersect(trainval, testval)

    output.remove('itemID_1')
    output.remove('itemID_2')

    # columns
    print(train.columns.values)
    print(test.columns.values)

    output.remove('Unnamed: 0_x')
    output.remove('Unnamed: 0_y')
    output.remove('Unnamed: 0.1')

    return output


def prep_train():
    testing = 0
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_train.csv")
    pairs = pd.read_csv(INPUT_DIR + "ItemPairs_train.csv", dtype=types1)

    print("Load ItemInfo_train.csv")
    items = pd.read_csv(INPUT_DIR + "ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv(INPUT_DIR + "Location.csv")
    category = pd.read_csv(INPUT_DIR + "Category.csv")

    train = pairs
    train = train.drop(['generationMethod'], axis=1)

    print('Add text features...')
    items['len_title'] = items['title'].str.len()
    items['len_description'] = items['description'].str.len()
    items['len_attrsJSON'] = items['attrsJSON'].str.len()

    print('Merge item 1...')
    item1 = items[[
        'itemID', 'categoryID', 'price', 'locationID',
        'metroID', 'lat', 'lon', 'len_title', 'len_description',
        'len_attrsJSON', 'title', 'description'
    ]]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1',
            'len_title': 'len_title_1',
            'len_description': 'len_description_1',
            'len_attrsJSON': 'len_attrsJSON_1',
            'title': 'title_1',
            'description': 'description_1',
            # 'images_array': 'images_array_1'
        }
    )

    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[[
        'itemID', 'categoryID', 'price', 'locationID',
        'metroID', 'lat', 'lon', 'len_title', 'len_description',
        'len_attrsJSON', 'title', 'description'
    ]]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2',
            'len_title': 'len_title_2',
            'len_description': 'len_description_2',
            'len_attrsJSON': 'len_attrsJSON_2',
            'title': 'title_2',
            'description': 'description_2',
            # 'images_array': 'images_array_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    # add text similarities
    # types_texts = {
    #     'itemID_1': np.dtype(int),
    #     'itemID_2': np.dtype(int),
    #     'isDuplicate': np.dtype(int),
    #     'generationMethod': np.dtype(int),
    #     'title_sim': np.dtype(float),
    #     'description_sim': np.dtype(float),
    # }

    # texts from w2vec
    texts_sim = pd.read_csv(INPUT_DIR + 'texts/splits/w2v_train.csv', compression='gzip')
    texts_sim.drop(['generationMethod', 'isDuplicate', 'Unnamed: 0.1.1', 'attrsJSON_x', 'attrsJSON_y'], axis=1, inplace=True)
    train = pd.merge(train, texts_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    # texts
    # texts_sim = pd.read_csv(INPUT_DIR + 'texts/FPairs_text_IDX_train.csv', compression='gzip')
    # texts_sim.drop(['generationMethod', 'isDuplicate', 'Unnamed: 0.1'], axis=1, inplace=True)
    # train = pd.merge(train, texts_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    # json_simO = pd.read_csv(INPUT_DIR + 'texts/FPairs_JSON_onlvidx_train.csv', compression='gzip')
    # json_simO.drop(['generationMethod', 'isDuplicate', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    # json_simO = json_simO.rename(columns={'json_sim': 'json_simO'})
    # train = pd.merge(train, json_simO, how='left', on=['itemID_2', 'itemID_1'])

    json_simK = pd.read_csv(INPUT_DIR + 'texts/FPairs_JSON_kidxvidx_train.csv', compression='gzip')
    json_simK.drop(['generationMethod', 'isDuplicate', 'Unnamed: 0.1', 'Unnamed: 0.1'], axis=1, inplace=True)
    train = pd.merge(train, json_simK, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    # image hashes
    img_sim = pd.read_csv(INPUT_DIR + 'all_train.csv', compression='gzip')
    img_sim.drop([
        'Unnamed: 0', 'isDuplicate', 'generationMethod', 'cat_same', 'pcat_same',
        'title', 'description',	'loc_same',	'rloc_same', 'metro_same',
        'pdiff', 'pavg'
    ], axis=1, inplace=True)
    train = pd.merge(train, img_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)
    # train['dhash'] = train.apply(lambda x: flowdist(train[x['images_array_1']], train[x['images_array_2']]), axis=1)
    # train['whash'] = train.apply(lambda x: flowdist2(train[x['images_array_1']], train[x['images_array_2']]), axis=1)

    # Create arrays
    print('Create same arrays')
    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['parentCategoryID_same'] = np.equal(train['parentCategoryID_1'], train['parentCategoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)
    train['len_title_same'] = np.equal(train['len_title_1'], train['len_title_2']).astype(np.int32)
    train['len_description_same'] = np.equal(train['len_description_1'], train['len_description_2']).astype(np.int32)
    train['len_attrsJSON_same'] = np.equal(train['len_attrsJSON_1'], train['len_attrsJSON_2']).astype(np.int32)

    # train['lat_diff'] = np.abs(train['lat_1'] - train['lat_2'])
    # train['lon_diff'] = np.abs(train['lon_1'] - train['lon_2'])

    lon1 = train['lon_1']
    lon2 = train['lon_2']
    lat1 = train['lat_1']
    lat2 = train['lat_2']
    train['dist'] = pow(pow(lon1 - lon2, 2) + pow(lat1 - lat2, 2), 0.5)

    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train


def prep_test():
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'id': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_test.csv")
    pairs = pd.read_csv(INPUT_DIR + "ItemPairs_test.csv", dtype=types1)

    print("Load ItemInfo_testcsv")
    items = pd.read_csv(INPUT_DIR + "ItemInfo_test.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv(INPUT_DIR + "Location.csv")
    category = pd.read_csv(INPUT_DIR + "Category.csv")

    train = pairs

    print('Add text features...')
    items['len_title'] = items['title'].str.len()
    items['len_description'] = items['description'].str.len()
    items['len_attrsJSON'] = items['attrsJSON'].str.len()

    print('Merge item 1...')
    item1 = items[[
        'itemID', 'categoryID', 'price', 'locationID',
        'metroID', 'lat', 'lon', 'len_title',
        'len_description', 'len_attrsJSON'
    ]]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1',
            'len_title': 'len_title_1',
            'len_description': 'len_description_1',
            'len_attrsJSON': 'len_attrsJSON_1'
        }
    )

    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[[
        'itemID', 'categoryID', 'price', 'locationID',
        'metroID', 'lat', 'lon', 'len_title',
        'len_description', 'len_attrsJSON'
    ]]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2',
            'len_title': 'len_title_2',
            'len_description': 'len_description_2',
            'len_attrsJSON': 'len_attrsJSON_2'
        }
    )

    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    # add text similarities
    # types_texts = {
    #     'itemID_1': np.dtype(int),
    #     'itemID_2': np.dtype(int),
    #     'title_sim': np.dtype(float),
    #     'description_sim': np.dtype(float),
    # }

    texts_sim = pd.read_csv(INPUT_DIR + 'texts/splits/w2v_test.csv', compression='gzip')
    texts_sim.drop(['attrsJSON_x', 'attrsJSON_y'], axis=1, inplace=True)
    train = pd.merge(train, texts_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    # texts_sim = pd.read_csv(INPUT_DIR + 'texts/FPairs_text_IDX_test.csv', compression='gzip')
    # texts_sim.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    # train = pd.merge(train, texts_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    # json_simO = pd.read_csv(INPUT_DIR + 'texts/FPairs_JSON_onlvidx_test.csv', compression='gzip')
    # json_simO.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
    # json_simO = json_simO.rename(columns={'json_sim': 'json_simO'})
    # train = pd.merge(train, json_simO, how='left', on=['itemID_2', 'itemID_1'])

    json_sim = pd.read_csv(INPUT_DIR + 'texts/FPairs_JSON_kidxvidx_test.csv', compression='gzip')
    json_sim.drop(['Unnamed: 0.1', 'id'], axis=1, inplace=True)
    train = pd.merge(train, json_sim, how='left', on=['itemID_2', 'itemID_1'])

    # image hashes
    img_sim = pd.read_csv(INPUT_DIR + 'all_test.csv', compression='gzip')
    img_sim.drop([
        'Unnamed: 0', 'cat_same', 'pcat_same',
        'title', 'description', 'loc_same', 'rloc_same', 'metro_same',
        'pdiff', 'pavg'
    ], axis=1, inplace=True)
    train = pd.merge(train, img_sim, how='left', on=['itemID_2', 'itemID_1'], left_index=True)

    #
    #
    #

    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['parentCategoryID_same'] = np.equal(train['parentCategoryID_1'], train['parentCategoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)
    train['len_title_same'] = np.equal(train['len_title_1'], train['len_title_2']).astype(np.int32)
    train['len_description_same'] = np.equal(train['len_description_1'], train['len_description_2']).astype(np.int32)
    train['len_attrsJSON_same'] = np.equal(train['len_attrsJSON_1'], train['len_attrsJSON_2']).astype(np.int32)

    # train['lat_diff'] = train.apply(lambda x: abs(train['lat_1'] - train['lat_2']), axis=1)
    # train['lon_diff'] = train.apply(lambda x: abs(train['lon_1'] - train['lon_2']), axis=1)
    # train['lat_diff'] = np.abs(train['lat_1'] - train['lat_2'])
    # train['lon_diff'] = np.abs(train['lon_1'] - train['lon_2'])

    lon1 = train['lon_1']
    lon2 = train['lon_2']
    lat1 = train['lat_1']
    lat2 = train['lat_2']
    train['dist'] = pow(pow(lon1 - lon2, 2) + pow(lat1 - lat2, 2), 0.5)

    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train


def read_test_train():

    train = prep_train()
    test = prep_test()

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    len_old = len(train.index)
    train = train.sample(frac=0.5)
    len_new = len(train.index)
    print('Reduce train from {} to {}'.format(len_old, len_new))

    features = get_features(train, test)

    return train, test, features

#
#
#

if __name__ == '__main__':

    train, test, features = read_test_train()
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    test_prediction, score = run_default_test(train, test, features, 'isDuplicate')
    print('Real score = {}'.format(score))

    create_submission(score, test, test_prediction)

# train-auc:0.974161	eval-auc:0.922967
# ('Importance array: ', [('description_sim', 53991), ('len_description_2', 47134), ('len_description_1', 45923), ('price_1', 39630), ('price_2', 38797), ('len_attrsJSON_1', 35283), ('title_sim', 33814), ('len_attrsJSON_2', 32220), ('len_title_2', 31451), ('len_title_1', 31106), ('locationID_2', 24269), ('dist', 20755), ('categoryID_2', 20168), ('locationID_1', 18455), ('metroID_1', 15535), ('json_sim', 15072), ('metroID_2', 14716), ('parentCategoryID_1', 9399), ('regionID_1', 5453), ('regionID_2', 4104), ('price_same', 4037), ('categoryID_1', 3871), ('parentCategoryID_2', 1988), ('locationID_same', 1437)])
