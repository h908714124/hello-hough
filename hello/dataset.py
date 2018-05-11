from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data.data_labels as dlbl
import data.data_images as dimg

import numpy as np


def dataset(labels='', images=''):

  labels = tf.data.Dataset.from_tensor_slices(dlbl.labels)
  images = tf.data.Dataset.from_tensor_slices(dimg.images)
  zipped = tf.data.Dataset.zip((images, labels))
  return zipped
