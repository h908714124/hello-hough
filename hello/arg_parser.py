import argparse


class ArgParser(argparse.ArgumentParser):
  def __init__(self):
    super(ArgParser, self).__init__()
    self.add_argument(
      'image', nargs='?'
    )
    self.add_argument(
      '--create_dataset', action="store_true"
    )
    self.add_argument(
      '--show_lines', action="store_true"
    )
    self.add_argument(
      '--read_dataset', action="store_true"
    )
    self.add_argument(
      '--dataset_images', default='/tmp/images'
    )
    self.add_argument(
      '--dataset_labels', default='/tmp/labels'
    )
    self.add_argument(
      '--train', action="store_true"
    )
    self.add_argument(
      '--predict', action="store_true"
    )
    self.add_argument(
      "--model_dir", default="/tmp/go_model"
    )
    self.add_argument(
      "--export_dir", default="trained_model"
    )
