import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn import preprocessing
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
TEST_IMAGES_DIR = './tiny-imagenet-200/test/'
TRAIN_SIZE = 10000
TEST_SIZE = 1000
CAT_SIZE = 100
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

class DataSet(object):
  def __init__(self,
               images,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end]

def input_data():
    trainingImages = load_training_images()
    testImages = load_test_images()
    options = dict(dtype=dtypes.float32, reshape=True, seed=None)
    train = DataSet(trainingImages, **options)
    test = DataSet(testImages, **options)
    return base.Datasets(train=train, validation=test, test=test)

def load_test_images():
    image_index = 0

    images = np.ndarray(shape=(TEST_SIZE, 64,64,3))

    print("Loading training images from ", TEST_IMAGES_DIR)
    # Loop through all the types directories
    for type in os.listdir(TEST_IMAGES_DIR):
        b_index = 0
        if os.path.isdir(TEST_IMAGES_DIR + type + '/images/'):
            type_images = os.listdir(TEST_IMAGES_DIR + type + '/images/')
            for image in type_images:
                image_file = os.path.join(TEST_IMAGES_DIR, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                # print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :,:,:] = image_data
                    image_index += 1
                    if image_index >= TEST_SIZE:
                        return images
                if b_index >= CAT_SIZE:
                    break

        if image_index >= TEST_SIZE:
                return images
    return images

def load_training_images():
    image_index = 0

    images = np.ndarray(shape=(TRAIN_SIZE, 64,64,3))

    print("Loading training images from ", TRAINING_IMAGES_DIR)
    # Loop through all the types directories
    for type in os.listdir(TRAINING_IMAGES_DIR):
        b_index = 0
        if os.path.isdir(TRAINING_IMAGES_DIR + type + '/images/'):
            type_images = os.listdir(TRAINING_IMAGES_DIR + type + '/images/')
            for image in type_images:
                image_file = os.path.join(TRAINING_IMAGES_DIR, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                # print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :,:,:] = image_data
                    image_index += 1
                    if image_index >= TRAIN_SIZE:
                        return images
                if b_index >= CAT_SIZE:
                    break

        if image_index >= TRAIN_SIZE:
            return images
    return images

