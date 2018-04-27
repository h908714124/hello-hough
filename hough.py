from __future__ import print_function

import glob
import os
import sys

import PIL
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

from hello.arg_parser import ArgParser
from hello.basic_data_ops import show_dataset, find_lines, get_tiles, show_lines, create_dataset
from hello.dataset import dataset
from hello.im_debug import show
from hello.model_def import cnn_model_fn


def load(input_file):
  img = PIL.Image.open(input_file).convert('L')
  img = img.resize([28, 28], PIL.Image.BILINEAR)
  m = np.zeros((28, 28), np.float32)
  for i in range(0, 28):
    for j in range(0, 28):
      m[i, j] = 0.00390625 * img.getpixel((j, i))
  show(m)
  return [m]


def predict(flags):
  if flags.image is None:
    raise Exception("Missing argument 'image'")
  if flags.export_dir is None:
    raise Exception("Missing argument 'export_dir'")
  list_of_files = glob.glob(flags.export_dir + '/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  predict_fn = predictor.from_saved_model(latest_file)
  predictions = predict_fn({'image': load(flags.image)})
  print('Prediction: %d' % (np.argmax(predictions['probabilities'][0])))


def train(flags):
  if flags.export_dir is None:
    raise Exception("Missing argument 'export_dir'")
  if flags.train_labels is None:
    raise Exception("Missing argument 'train_labels'")
  if flags.train_images is None:
    raise Exception("Missing argument 'train_images'")
  if not os.path.isfile(flags.train_labels):
    raise Exception("Training labels don't exist. Run --train first.")
  if not os.path.isfile(flags.train_images):
    raise Exception("Training images don't exist. Run --train first.")
  
  def train_input_fn():
    ds = dataset(labels=flags.train_labels, images=flags.train_images)
    ds = ds.cache().shuffle(buffer_size=50000).batch(10)
    return ds
  
  go_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=flags.model_dir)
  
  go_classifier.train(input_fn=train_input_fn)
  
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
