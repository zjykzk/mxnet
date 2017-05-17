from __future__ import print_function
import cPickle
import cv2
import os
import numpy as np
import collections

from imdb import IMDB
from imagenet_loc_2017 import imagenet_loc_2017


class imagenet_loc_val_2017(imagenet_loc_2017):
    def __init__(self, image_set, root_path, data_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param data_path: data and results
        :return: imdb object
        """
        super(imagenet_loc_val_2017, self).__init__(image_set,
             root_path, data_path)
        self.name = 'imagenet_loc_val_2017_' + image_set

    def _get_index_file(self):
        return os.path.join(self.root_path, self.image_set + '.txt')
