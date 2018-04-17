import argparse


class ArgParser(argparse.ArgumentParser):
  def __init__(self):
    super(ArgParser, self).__init__()
    self.add_argument(
      "--image", "-i", required=True
    )
