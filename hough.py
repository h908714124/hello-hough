from __future__ import print_function

import sys

import cv2
import numpy as np
import tensorflow as tf

from hello.arg_parser import ArgParser
from hello.im_debug import show

tf.enable_eager_execution(config=None, device_policy=None)


def main(flags):
  img = cv2.imread(flags.image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
  lines = tf.reshape(lines, [-1, 2]).numpy()
  
  for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    factor = 500
    x1 = int(x0 + factor * (-b))
    y1 = int(y0 + factor * (a))
    x2 = int(x0 - factor * (-b))
    y2 = int(y0 - factor * (a))
    
    cv2.line(img, (x1, y1), (x2, y2), (245, 0, 0), 2)
  
  show(img)


if __name__ == '__main__':
  flags = ArgParser().parse_args(args=sys.argv[1:])
  main(flags)
