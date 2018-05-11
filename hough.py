from __future__ import print_function

import glob
import os
import sys

import cv2
import tensorflow as tf
from tensorflow.contrib import predictor

import data.data_images as dimg
import numpy as np

from hello.arg_parser import ArgParser
from hello.basic_data_ops import show_dataset, find_lines, get_tiles, show_lines, create_dataset
from hello.dataset import dataset
from hello.model_def import cnn_model_fn


def load(i):
  return [dimg.images[i]]


def predict(flags):
  if flags.image is None:
    raise Exception("Missing argument 'image'")
  if flags.export_dir is None:
    raise Exception("Missing argument 'export_dir'")
  list_of_files = glob.glob(flags.export_dir + '/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  predict_fn = predictor.from_saved_model(latest_file)
  for n in range(len(dimg.images)):
    image = load(n)
    predictions = predict_fn({'image': image})
    print('image: %03s Prediction: %s' % (n, np.argmax(predictions['probabilities'][0])))
    # print(np.reshape(image, [-1]))


def train(flags):
  if flags.export_dir is None:
    raise Exception("Missing argument 'export_dir'")
  if flags.train_labels is None:
    raise Exception("Missing argument 'train_labels'")
  if flags.train_images is None:
    raise Exception("Missing argument 'train_images'")
  if not os.path.isfile(flags.train_labels):
    raise Exception("Training labels don't exist. Run --create_dataset first.")
  if not os.path.isfile(flags.train_images):
    raise Exception("Training images don't exist. Run --create_dataset first.")

  def train_input_fn():
    ds = dataset(labels=flags.train_labels, images=flags.train_images)
    ds = ds.cache().shuffle(buffer_size=50).batch(12)
    return ds

  def eval_input_fn():
    return dataset(labels=flags.train_labels, images=flags.train_images).batch(
      12).shuffle(buffer_size=50).make_one_shot_iterator().get_next()

  go_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=flags.model_dir)

  for _ in range(50):
    go_classifier.train(input_fn=train_input_fn)
    eval_results = go_classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

  image = tf.placeholder(tf.float32, [None, 784])
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
    train(flags)
    return

  if flags.read_dataset:
    show_dataset(flags.train_images)
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
    create_dataset(tiles, flags)

  if flags.show_lines:
    show_lines(img, horizontal_lines, vertical_lines)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
