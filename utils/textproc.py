# -*- coding: utf-8 -*-
import numpy as np
from rutermextract import TermExtractor
from fuzzywuzzy import fuzz


def get_similarity(arg1, arg2):

    term_extractor = TermExtractor()

    # try:
    #     subterms1 = term_extractor(arg1, nested=True)
    # except TypeError as exep:
    #     print exep.args
    #     x = exep
    #     print 'arg1 = ', x

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

    #print "ratio: %f" % ratio
    # for term0 in intersection:
    #     print "intersection %s" % term0.normalized

    set1_ = set.symmetric_difference(set1, intersection)
    set2_ = set.symmetric_difference(set2, intersection)

    for term1 in set1_:
        for term2 in set2_:
            # rat = fuzz.ratio(term1.normalized, term2.normalized)
            rat = fuzz.partial_ratio(term1.normalized, term2.normalized)
            # print rat
            if rat > 30:
                ratio += rat*0.01

    metric = ratio/average_length   # TODO: mean or smth else
    # metric = np.mean(ratio)
    # print "similarity: %f" % metric

    return metric


# get_similarity(
#     u"продаю ботинки сноубордические размер 42,5 10 us Nitro Team 28см, возможен торг",
#     u"ботинки продаю новые ботинки 42,5 10 us Nitro Team 28см, самовывоз из Твери"
# )
#
# get_similarity(
#     u"продаю ботинки сноубордические размер 42,5 10 us Nitro Team 28см, возможен торг",
#     u"Весь в тюнинге. Возможен торг"
# )
