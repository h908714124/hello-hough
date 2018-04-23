from __future__ import print_function

import sys

import PIL.Image
import cv2
import numpy as np
import tensorflow as tf

from hello.arg_parser import ArgParser
from hello.im_debug import show

grid_size = 19

tf.enable_eager_execution(config=None, device_policy=None)


def add_horizontal_line(img, y):
  width = np.shape(img)[1]
  cv2.line(img, (0, y), (width, y), (245, 0, 0), 1)


def add_vertical_line(img, x):
  height = np.shape(img)[0]
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
  mean_diff = np.mean(diff)
  deviations = [np.square(mean_diff - x) for x in diff]
  cutoff = np.max(deviations) * 0.375
  result = np.zeros(len(lines))
  result_pos = 0
  for index, line in enumerate(lines):
    if index == 0:
      d = np.square(lines[index + 1] - line - mean_diff)
      if d < cutoff:
        result[result_pos] = line
        result_pos += 1
    elif index == len(lines) - 1:
      d = np.square(lines[index - 1] - line - mean_diff)
      if d < cutoff:
        result[result_pos] = line
        result_pos += 1
    else:
      left = np.square(line - lines[index - 1] - mean_diff)
      right = np.square(lines[index + 1] - line - mean_diff)
      if left < cutoff or right < cutoff:
        result[result_pos] = line
        result_pos += 1
  result = result[0:result_pos]
  if len(result) != grid_size:
    raise Exception("Can't find the grid!")
  return result


def pad_left(img, x1):
  return np.pad(img, ((0, 0), (abs(x1), 0)), 'edge')


def pad_right(img, x2):
  w = np.shape(img)[1]
  return np.pad(img, ((0, 0), (0, abs(x2 - w))), 'edge')


def pad_top(img, y1):
  return np.pad(img, ((abs(y1), 0), (0, 0)), 'edge')


def pad_bottom(img, y2):
  h = np.shape(img)[0]
  return np.pad(img, (0, (abs(y2 - h)), (0, 0)), 'edge')


def initial_tile(img, x1, x2, y1, y2):
  h = np.shape(img)[0]
  w = np.shape(img)[1]
  y1 = max(y1, 0)
  y2 = min(y2, h)
  x1 = max(x1, 0)
  x2 = min(x2, w)
  return img[y1:y2, x1:x2]


# 4x4x3 matrix: A = np.arange(48).reshape(-1,3,4)
def get_tiles(img, horizontal_lines, vertical_lines):
  hdiff = np.diff(horizontal_lines)
  mean_hdiff = np.mean(hdiff)
  vdiff = np.diff(vertical_lines)
  mean_vdiff = np.mean(vdiff)
  result = np.zeros([361, 32, 32], dtype=np.float32)
  result_pos = 0
  h = np.shape(img)[0]
  w = np.shape(img)[1]
  for hline in iter(horizontal_lines):
    for vline in iter(vertical_lines):
      y1 = int(hline - (mean_hdiff / 2))
      y2 = int(hline + (mean_hdiff / 2))
      x1 = int(vline - (mean_vdiff / 2))
      x2 = int(vline + (mean_vdiff / 2))
      tile = initial_tile(img, x1, x2, y1, y2)
      if (x1 < 0):
        tile = pad_left(img, x1)
      if (x2 > w):
        tile = pad_right(img, x2)
      if (y1 < 0):
        tile = pad_top(img, y1)
      if (y2 > h):
        tile = pad_bottom(img, y2)
      pil_image = PIL.Image.fromarray(tile)
      pil_image = pil_image.resize([32, 32], PIL.Image.BILINEAR)
      result[result_pos] = np.asarray(pil_image, dtype=np.float32)
      result_pos += 1
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
  
  tiles = get_tiles(gray, horizontal_lines, vertical_lines)
  show(tiles[0])
  show(tiles[60])
  
  for y in horizontal_lines:
    add_horizontal_line(img, int(y))
  
  for x in vertical_lines:
    add_vertical_line(img, int(x))
  
  show(img)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
