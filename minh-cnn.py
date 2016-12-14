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
NUM_LABELS = 10 # The number of detectable objects + 1 background label
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
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
        self.train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
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
        # Channelwise inhibited

    def extract_data(filepath):
        """Extract the images into a tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        filenames = glob.iglob(filepath)
        for filename in filenames:
            print('Extracting', filename)
            with Image.open(filename) as input_img:
                input_img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_data = numpy.array(input_img.getdata(),
                                       numpy.uint8).reshape(input_img.size[1], input_img.size[0], 3)
                img_data = (img_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

        img_data = img_data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return img_data

    def extract_labels(filepath):
        """Extract the labels into a vector of int64 label IDs."""
        filenames = glob.iglob(filepath)
        print('Extracting', filepath)
        for filename in filenames:
            with open(filename) as lablefile:
                pass
        return None

    def model(self, data, train=False):
        """The Model definition."""
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
        fc1 = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5, seed=SEED)
        fc2 = tf.matmul(fc1, self.fc2_weights) + self.fc2_biases
        return fc2

    def train(self):
        # Visit the data directories
        training_filenames = glob.glob(TRAINING_DIRECTORY).sort()
        annotation_filenames = glob.glob(ANNOTATION_DIRECTORY).sort()
        assert(len(training_filenames) == len(annotation_filenames))
        train_size = len(training_filenames) - VALIDATION_SIZE

        # Training computation: logits + cross-entropy loss.
        logits = self.model(self.train_data_node, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers #TODO edit this

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, dtype=data_type())
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
                0.01,                # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                train_size,          # Decay step.
                0.95,                # Decay rate.
                staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                0.9).minimize(loss,
                        global_step=batch)

                # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)

        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(self.model(self.eval_data))

        # Create a local session to run the training.
        start_time = time.time()
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()
            print('Initialized!')
        # Loop through training steps.
        for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

            # Load the data and labels TODO
            batch_data = training_filenames[offset:(offset + BATCH_SIZE), ...]
            batch_labels = annotation_filenames[offset:(offset + BATCH_SIZE)]

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {self.train_data_node: batch_data,
                        self.train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                        feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                    (step, float(step) * BATCH_SIZE / train_size,
                        1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    self.eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()

    def eval_in_batches(self, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        # Small utility function to evaluate a dataset by feeding batches of data to
        # {self.eval_data} and pulling the results from {eval_predictions}.
        # Saves memory and enables this to run on smaller GPUs.
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
                raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={self.eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                    eval_prediction, feed_dict={self.eval_data: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

def main(_):
    ann_frame = pd.read_csv(ANNOTATION_DIRECTORY + ANNOTATION_MAIN, sep=r' ')
    for fold_num in xrange(len(FOLDS)):
        # Start the fold
        with open(ANNOTATION_DIRECTORY + FOLDS[fold_num]) as an_file:
            fold_filenames = list(map(lambda x: x.strip(), an_file.readlines()))

        num_batches = ceil(len(fold_filenames) / flaot(BATCH_SIZE))
        num_images = 0
        if len(fold_filenames) >= BATCH_SIZE:
            batch_list = [ None for i in xrange(BATCH_SIZE) ]
        else:
            batch_list = [ None for i in xrange(len(fold_filenames)]
        for fold_filename in fold_filenames:
            if num_images >= BATCH_SIZE:
                batch_ind += 1
            filename_list[batch_ind].append
            num_images += 1
            training_img = Image.read(TRAINING_DIRECTORY + fold_filename + IMG_EXT)
            training_img.resize((IMAGE_SIZE, IMAGE_SIZE))
            training_array = numpy.array(training_img.getdata(),
                    numpy.uint8).reshape(training_img.size[1], training_img.size[0], 3)
            image_list[batch_ind].append(training_array)
            num_images += 1


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
