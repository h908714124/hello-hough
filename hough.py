from __future__ import print_function

import os
import sys

import cv2
import tensorflow as tf

from hello.arg_parser import ArgParser
from hello.basic_data_ops import show_dataset, find_lines, get_tiles, show_lines, create_dataset

tf.enable_eager_execution(config=None, device_policy=None)


def main(flags):
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
