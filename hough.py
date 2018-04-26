from __future__ import print_function

import os
import sys
import glob

import cv2
import tensorflow as tf
import numpy as np

from hello.dataset import dataset

from hello.arg_parser import ArgParser
from hello.model_def import cnn_model_fn
from hello.basic_data_ops import show_dataset, find_lines, get_tiles, show_lines, create_dataset
from tensorflow.contrib import predictor


def predict(flags):
  list_of_files = glob.glob(flags.export_dir + '/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  predict_fn = predictor.from_saved_model(latest_file)
  predictions = predict_fn({'image': load(flags.input_file)})
  print('Prediction: %d' % (np.argmax(predictions['probabilities'][0])))


def train(labels='', images=''):
  def train_input_fn():
    ds = dataset(labels=labels, images=images)
    ds = ds.cache().shuffle(buffer_size=50000).batch(10)
    return ds
  
  go_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=flags.model_dir)
  
  go_classifier.train(input_fn=train_input_fn)
  
  if flags.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      'image': image,
    })
    export_dir = go_classifier.export_savedmodel(flags.export_dir, input_fn)
    print('\nExported to:\n\t%s\n' % export_dir)


def main(flags):
  if flags.predict:
    predict(flags)
    return
  
  if flags.train:
    train(labels=flags.dataset_labels, images=flags.dataset_images)
    return
  
  if flags.read_dataset:
    show_dataset(flags.dataset_images)
    return
  
  if flags.image is None:
    raise Exception("Missing argument 'image'")
  
  if os.path.isfile(flags.image):
    img = cv2.imread(flags.image)
  else:
    raise ValueError('Path provided is not a valid file: {}'.format(flags.image))
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  horizontal_lines, vertical_lines = find_lines(img, gray)
  tiles = get_tiles(gray, horizontal_lines, vertical_lines)
  
  if flags.create_dataset:
    create_dataset(tiles, labels=flags.dataset_labels, images=flags.dataset_images)
  
  if flags.show_lines:
    show_lines(img, horizontal_lines, vertical_lines)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
