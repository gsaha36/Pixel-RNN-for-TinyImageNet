import logging
logging.basicConfig(format="[%(asctime)s] %(filename)s [line:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S")

import os
import sys
import urllib
import pprint
import tarfile
import tensorflow as tf
import json
import datetime
import dateutil.tz
import numpy as np

import scipy.misc

pp = pprint.PrettyPrinter().pprint
logger = logging.getLogger(__name__)

def setup_model_saving(model_name, data, hyperparams=None, root_dir='run/'):
  # construct the model directory template name
  name = os.path.join(root_dir, data, model_name + '%s')
  # iterate until we find an index that hasn't been taken yet.
  i = 0
  while os.path.exists(name % i):
    i += 1
  name = name % i
  # create the folder
  os.makedirs(name)
  return name

def mprint(matrix, pivot=0.5):
  for array in matrix:
    print("".join("#" if i > pivot else " " for i in array))

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
    total_count += int(count)
  print("[Total] variable size: %s" % "{:,}".format(total_count))

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')

def binarize(images):
  return (np.random.uniform(size=images.shape) < images).astype('float32')

def save_images_in(images,cmin=0.0, cmax=1.0,directory="./",prefix="sample"):
  index = 1
  for i in np.arange(images.shape[0]):
    filename = '%s_%s_%s.jpg' % (i,prefix, get_timestamp())
    scipy.misc.toimage(images[i].reshape(images[i].shape[0],images[i].shape[1]), cmin=0, cmax=255).save(filename)

def save_images(images, height, width, n_row, n_col, cmin=0.0, cmax=1.0, directory="./", prefix="sample"):
  images = images.reshape((n_row, n_col, height, width))
  images = images.transpose(1, 2, 0, 3)
  images = images.reshape((height * n_row, width * n_col))

  filename = '%s_%s.jpg' % (prefix, get_timestamp())
  scipy.misc.toimage(images, cmin=cmin, cmax=cmax).save(os.path.join(directory, filename))

def get_model_dir(conf, exceptions=None):
  # attrs = conf.__dict__['__flags']
  # pp(attrs)

  keys = conf.flag_values_dict()
  # keys.remove('data')
  # keys = ['data'] + keys

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in conf[key]])
          if type(conf[key]) == list else conf[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags

  for option, value in options.items():
    option = option.lower()

def check_and_create_dir(directory):
  if not os.path.exists(directory):
    logger.info('Creating directory: %s' % directory)
    os.makedirs(directory)
  else:
    logger.info('Skip creating directory: %s' % directory)

def maybe_download_and_extract(dest_directory):
  """
  Download and extract the tarball from Alex's website.
  From https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/cifar10/cifar10.py
  """
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
