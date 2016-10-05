# -*- coding: utf-8 -*-
import os


def get_img_path(img_id, data_path):

    if img_id < 10:
        folder = str(img_id)[-1:]
    else:
        folder = str(img_id)[-2:]

    if folder[0] == '0':
        folder = folder[1]

    path = os.path.join(data_path, folder, str(img_id) + '.jpg')

    return path
