from __future__ import print_function

import sys

import cv2
import numpy as np
import tensorflow as tf

from hello.arg_parser import ArgParser
from hello.im_debug import show

tf.enable_eager_execution(config=None, device_policy=None)


def add_horizontal_line(img, rho):
  factor = 500
  x1 = int(-factor)
  y = int(rho)
  x2 = int(factor)
  cv2.line(img, (x1, y), (x2, y), (245, 0, 0), 1)


def add_vertical_line(img, rho):
  factor = 500
  x = int(rho)
  y1 = int(factor)
  y2 = int(- factor)
  cv2.line(img, (x, y1), (x, y2), (0, 245, 0), 1)


def split_lines(lines):
  horizontal_lines = np.zeros(np.shape(lines)[0])
  vertical_lines = np.zeros(np.shape(lines)[0])
  for index, x in enumerate(lines):
    print("x:", x)
    print("index:", index)
    if x[1] < 0.5:
      horizontal_lines[index] = x[0]
    else:
      vertical_lines[index] = x[0]
  return horizontal_lines, vertical_lines


def main(flags):
  img = cv2.imread(flags.image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
  lines = tf.reshape(lines, [-1, 2]).numpy()
  
  horizontal_lines, vertical_lines = split_lines(lines)
  
  for rho in horizontal_lines:
    add_horizontal_line(img, rho)
  
  for rho in vertical_lines:
    add_vertical_line(img, rho)
  
  show(img)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
