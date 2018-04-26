from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def dataset(labels='', images=''):
  def decode_image(image):
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0
  
  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)
  
  images = tf.data.FixedLengthRecordDataset(
    images, 28 * 28, header_bytes=0).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
    labels, 1, header_bytes=0).map(decode_label)
  return tf.data.Dataset.zip((images, labels))
