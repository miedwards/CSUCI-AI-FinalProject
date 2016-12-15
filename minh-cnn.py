# Modified by Mark Edwards based on code by The TensorFlow Authors.
# See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image/mnist
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional model for object extraction
from aerial traffic photos
"""
import argparse
import os
import sys
import time
import glob
from PIL import Image
import pandas as pd
import numpy
from six.moves import xrange
import tensorflow as tf
from math import ceil
from collections import defaultdict

WORK_DIRECTORY = '/Volumes/Seagate Expansion Drive/image_processing_project/aerial_image_data/vedai'
TRAINING_DIRECTORY = \
    r'/Volumes/Seagate Expansion Drive/image_processing_project/aerial_image_data/vedai/Vehicules512/'
ANNOTATION_DIRECTORY = \
    r'/Volumes/Seagate Expansion Drive/image_processing_project/aerial_image_data/vedai/Annotations512/'
IMAGE_SIZE = 512
OUTPUT_RESOLUTION = 16
NUM_CHANNELS = 3 # Number of color channels
PIXEL_DEPTH = 255 # Color depth
NUM_LABELS = 2 # The number of detectable objects + 1 background label
# The vedai has a maximum of 11 detectable objects ranging from 1 to 31,
# but this only locates for now. That is, it groups them all into 1 catagory
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
FC_DROPOUT = 0.5
IMG_EXT = r'_co.png'
ANN_EXT = r'.txt'
ANNOTATION_MAIN = r'annotation512.txt'
FOLDS = [ r'fold01.txt', r'fold02.txt', r'fold03.txt', r'fold04.txt',
          r'fold05.txt', r'fold06.txt', r'fold07.txt', r'fold08.txt',
          r'fold09.txt', r'fold10.txt' ]
FOLD_TESTS = [ r'fold01test.txt', r'fold02test.txt', r'fold03test.txt',
               r'fold04test.txt', r'fold05test.txt', r'fold06test.txt',
               r'fold07test.txt', r'fold08test.txt', r'fold09test.txt',
               r'fold10test.txt' ]
FLAGS = None

def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
      return tf.float16
    else:
      return tf.float32

def np_data_type():
    "Return the numpy data type."
    if FLAGS.use_fp16:
        return numpy.float16
    else:
        return numpy.float32

def fake_data(num_images):
    """Generate a fake dataset that matches the preset dimensions."""
    # FIXME
    data = numpy.ndarray(
            shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
            dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
    for image in xrange(num_images):
        label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
    return data, labels


def error_rate(predictions, labels): # TODO edit this
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == labels) /
            predictions.shape[0])


class MnihCNN: # Named for Volodymyr Mnih who developed this architecture
    def __init__(self):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.train_data_node = tf.placeholder(
                data_type(),
                shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int64,
                                shape=(BATCH_SIZE,
                                       OUTPUT_RESOLUTION,
                                       OUTPUT_RESOLUTION,
                                       NUM_LABELS))
        self.eval_data = tf.placeholder(
                data_type(),
                shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.global_variables_initializer().run()}
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([16, 16, NUM_CHANNELS, 64],  # 16x16 filter, depth 64.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
        self.conv1_biases = tf.Variable(tf.zeros([64], dtype=data_type()))
        self.conv2_weights = tf.Variable(tf.truncated_normal( # 4x4 filter depth 112
                                        [4, 4, 64, 112], stddev=0.1,
                                        seed=SEED, dtype=data_type()))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[112], dtype=data_type()))
        self.conv3_weights = tf.Variable(
            tf.truncated_normal([3, 3, 112, 80],  # 3x3 filter, depth 80.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
        self.conv3_biases = tf.Variable(tf.zeros([80], dtype=data_type()))
        num_output_neurons = OUTPUT_RESOLUTION * OUTPUT_RESOLUTION * NUM_LABELS
        fc_length = 16 * OUTPUT_RESOLUTION * OUTPUT_RESOLUTION
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
                                    tf.truncated_normal(
                                    [80 * IMAGE_SIZE * IMAGE_SIZE,
                                    fc_length],
                                    stddev=0.1,
                                    seed=SEED,
                                    dtype=data_type()))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc_length], dtype=data_type()))
        self.fc2_weights = tf.Variable(tf.truncated_normal([fc_length, num_output_neurons],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
        self.fc2_biases = tf.Variable(tf.constant(
            0.1, shape=[num_output_neurons], dtype=data_type()))
        # Minh net can be improved using channel inhibeting, but
        # this is not yet implemented.
        # Training computation: logits + cross-entropy loss.
        self.logits = self.model(self.train_data_node, True)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    self.logits, self.train_labels_node))
        # L2 regularization for the fully connected parameters.
        self.regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                            tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        self.batch = tf.Variable(0, dtype=data_type())
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            self.batch * BATCH_SIZE,  # Current index into the dataset.
            1100,                # Decay step. TODO FIXME
            0.95,                # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                0.9).minimize(self.loss,
                                                    global_step=self.batch)
        # Predictions for the current training minibatch.
        self.train_prediction = tf.nn.softmax(self.logits)
        # Predictions for the test and validation, which we'll compute less often.
        self.eval_prediction = tf.nn.softmax(model(self.eval_data))

    def model(self, data, train=False):
        """The Model definition."""
        print(data.get_shape())
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv1 = tf.nn.conv2d(data,
                self.conv1_weights,
                strides=[4, 4, 1, 1], # Bcause of our large first kernel, we need a bigger stride
                padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME')
        conv2 = tf.nn.conv2d(pool1,
                self.conv2_weights,
                strides=[1, 1, 1, 1],
                padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
        conv3 = tf.nn.conv2d(relu,
                self.conv3_weights,
                strides=[1, 1, 1, 1],
                padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_biases))
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        relu_shape = relu.get_shape().as_list()
        reshape = tf.reshape(
                relu,
                [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        print("Pre fc1")
        print(reshape.get_shape())
        print(self.fc1_weights.get_shape())
        fc1 = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        print("Post fc1")
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5, seed=SEED)
        print("Pre fc2")
        fc2 = tf.matmul(fc1, self.fc2_weights) + self.fc2_biases
        print("Post fc2")
        return fc2

    def get_empty_batchlist(num_total_files, num_files):
        if( num_total_files - num_files >= BATCH_SIZE ):
            return [ None for x in xrange(BATCH_SIZE) ]
        else:
            return [ None for x in xrange(num_total_files-num_files) ]

    def train(self):
        # Visit the annotation directories
        print("Reading annotations.")
        ann_frame = pd.read_csv(ANNOTATION_DIRECTORY + ANNOTATION_MAIN, sep=r' ')
        ann_lookup = {}
        for idx, img_num, center_x, center_y, angle, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, catagory, occluded, fully_included, in first_ann.itertuples():
            if(img_num not in ann_lookup):
                # Initialize the output to all background
                ann_lookup[img_num] = numpy.zeros(shape=(OUTPUT_RESOLUTION,
                                                         OUTPUT_RESOLUTION,
                                                         NUM_LABELS),
                                                  dtype=np_data_type())
                ann_lookup[img_num][:,:,0] = 1.0

            # Reset the value at this point using 1-hot encoding
            ann_lookup[img_num][center_x // OUTPUT_RESOLUTION,
                                center_y // OUTPUT_RESOLUTION,
                                0 ] = 0.0
            # Currently only 1 non-background label is supported
            # This is simple to change however
            ann_lookup[img_num][center_x // OUTPUT_RESOLUTION,
                                center_y // OUTPUT_RESOLUTION,
                                1 ] = 1.0
        with open(ANNOTATION_DIRECTORY + FOLDS[0],'r') as fold_file:
            fold_filenames = [ x.strip() for x in foldfile.readlines() ]

        with open(ANNOTATION_DIRECTORY + FOLD_TESTS[0], 'r') as test_file:
            test_filenames = [ x.strip() for x in test_filenames.readlines() ]

        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()
            print('Initialized!')
            num_batches = ceil(len(fold_filenames) / BATCH_SIZE)
            num_images = 0
            batch_ind = 0
            batch_offset = 0
            batch_list = [ None for x in xrange(BATCH_SIZE) ]
            for fold_filename in fold_filenames:
                if batch_offset == BATCH_SIZE:
                    # Prepare training images
                    batch_data = numpy.zeros((BATCH_SIZE,
                                              IMAGE_SIZE,
                                              IMAGE_SIZE,
                                              NUM_CHANNELS))
                    batch_labels = numpy.zeros((BATCH_SIZE,
                                              OUTPUT_RESOLUTION,
                                              OUTPUT_RESOLUTION,
                                              NUM_LABELS))
                    for in_filename_ind in xrange(len(batch_list)):
                        with Image.open(batch_list[in_filename_ind]) as input_img:
                            input_img.resize((IMAGE_SIZE, IMAGE_SIZE))
                            batch_data[in_filename_ind, :, :, :] = numpy.array(input_img.getdata(),
                                                                    ).reshape(input_img.size[1],
                                                                    input_img.size[0], NUM_CHANNELS)

                        batch_labels[in_filename_ind, :, :, :] = ann_lookup[in_filename_ind]

                    batch_data = (batch_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
                    feed_dict = {self.train_data_node: batch_data,
                                 self.train_labels_node: batch_labels}
                    print("Batch {} loaded".format(batch_ind))
                    # Run the optimizer to update weights.
                    sess.run(self.optimizer, feed_dict=feed_dict)
                    batch_ind += 1
                    batch_offset = 0
                    batch_list = [ None for x in xrange(BATCH_SIZE) ]

                batch_list[batch_offset] = fold_filename
                num_images += 1
                batch_offset += 1

def main(_):
    mcnn = MnihCNN()
    mcnn.train()


# test_error = error_rate(eval_in_batches(test_data, sess), test_labels) FIXME
# print('Test error: %.1f%%' % test_error)
# if FLAGS.self_test:
#     print('test_error', test_error)
# assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
#    test_error,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
                '--use_fp16',
                default=False,
                help='Use half floats instead of full floats if True.',
                action='store_true')
    parser.add_argument(
                '--self_test',
                default=False,
                action='store_true',
                help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
