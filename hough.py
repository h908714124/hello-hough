from __future__ import print_function

import sys

import cv2
import numpy as np
import tensorflow as tf

from hello.arg_parser import ArgParser
from hello.im_debug import show

grid_size = 19

tf.enable_eager_execution(config=None, device_policy=None)


def add_horizontal_line(img, y):
  width = np.shape(img)[0]
  cv2.line(img, (0, y), (width, y), (245, 0, 0), 1)


def add_vertical_line(img, x):
  height = np.shape(img)[1]
  cv2.line(img, (x, 0), (x, height), (0, 245, 0), 1)


def remove_duplicate_lines(max, lines):
  result = np.zeros(len(lines))
  i = 0
  result_pos = 0
  min_dist = max / 60
  for index, line in enumerate(lines):
    if line - lines[i] < min_dist:
      result[result_pos] = (line - lines[i]) / 2
    else:
      result[result_pos] = line
      result_pos += 1
      i = index
  result = result[0:result_pos]
  if len(result) < grid_size:
    raise Exception("Can't find the grid!")
  return result[0:result_pos]


def remove_outliers(lines):
  diff = np.diff(lines)
  mean = np.mean(diff)
  deviations = [np.square(mean - x) for x in diff]
  cutoff = np.max(deviations) / 2
  result = np.zeros(len(lines))
  result_pos = 0
  for index, line in enumerate(lines):
    if index == 0:
      d = np.square(lines[index + 1] - line - mean)
      if d < cutoff:
        result[result_pos] = line
        result_pos += 1
    elif index == len(lines) - 1:
      d = np.square(lines[index - 1] - line - mean)
      if d < cutoff:
        result[result_pos] = line
        result_pos += 1
    else:
      left = np.square(line - lines[index - 1] - mean)
      right = np.square(lines[index + 1] - line - mean)
      if left < cutoff or right < cutoff:
        result[result_pos] = line
        result_pos += 1
  result = result[0:result_pos]
  if len(result) != grid_size:
    raise Exception("Can't find the grid!")
  return result


def group_lines(lines):
  horizontal_lines = np.zeros(len(lines))
  vertical_lines = np.zeros(len(lines))
  i_h = 0
  i_v = 0
  for index, x in enumerate(lines):
    rho = x[0]
    theta = x[1]
    if theta < 0.75:  # about pi / 4 (45 degrees)
      horizontal_lines[i_h] = rho
      i_h += 1
    else:
      vertical_lines[i_v] = rho
      i_v += 1
  if i_v < grid_size or i_h < grid_size:
    raise Exception("Can't find the grid!")
  horizontal_lines = horizontal_lines[0:i_h]
  vertical_lines = vertical_lines[0:i_v]
  horizontal_lines = np.sort(horizontal_lines)
  vertical_lines = np.sort(vertical_lines)
  return horizontal_lines, vertical_lines


def main(flags):
  img = cv2.imread(flags.image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
  lines = np.reshape(lines, [-1, 2])
  
  horizontal_lines, vertical_lines = group_lines(lines)
  vertical_lines = remove_duplicate_lines(np.shape(img)[0], vertical_lines)
  horizontal_lines = remove_duplicate_lines(np.shape(img)[1], horizontal_lines)
  vertical_lines = remove_outliers(vertical_lines)
  horizontal_lines = remove_outliers(horizontal_lines)
  
  for y in horizontal_lines:
    add_horizontal_line(img, int(y))
  
  for x in vertical_lines:
    add_vertical_line(img, int(x))
  
  show(img)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
