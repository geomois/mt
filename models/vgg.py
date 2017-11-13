from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

VGG_FILE = './pretrained_params/vgg16_weights.npz'

def load_pretrained_VGG16_pool5(input, scope_name='vgg'):
    """
    Load an existing pretrained VGG-16 model.
    See https://www.cs.toronto.edu/~frossard/post/vgg16/

    Args:
        input:         4D Tensor, Input data
        scope_name:    Variable scope name

    Returns:
        pool5: 4D Tensor, last pooling layer
        assign_ops: List of TF operations, these operations assign pre-trained values
                    to all parameters.
    """

    with tf.variable_scope(scope_name):

        vgg_weights, vgg_keys = load_weights(VGG_FILE)

        assign_ops = []
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            vgg_W = vgg_weights['conv1_1_W']
            vgg_B = vgg_weights['conv1_1_b']
            kernel = tf.get_variable('conv1_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_1/' + "biases", vgg_B.shape,
                initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)


        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            vgg_W = vgg_weights['conv1_2_W']
            vgg_B = vgg_weights['conv1_2_b']
            kernel = tf.get_variable('conv1_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))

            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            vgg_W = vgg_weights['conv2_1_W']
            vgg_B = vgg_weights['conv2_1_b']
            kernel = tf.get_variable('conv2_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            vgg_W = vgg_weights['conv2_2_W']
            vgg_B = vgg_weights['conv2_2_b']
            kernel = tf.get_variable('conv2_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            vgg_W = vgg_weights['conv3_1_W']
            vgg_B = vgg_weights['conv3_1_b']
            kernel = tf.get_variable('conv3_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            vgg_W = vgg_weights['conv3_2_W']
            vgg_B = vgg_weights['conv3_2_b']
            kernel = tf.get_variable('conv3_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer()
                                     )

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            vgg_W = vgg_weights['conv3_3_W']
            vgg_B = vgg_weights['conv3_3_b']
            kernel = tf.get_variable('conv3_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            vgg_W = vgg_weights['conv4_1_W']
            vgg_B = vgg_weights['conv4_1_b']
            kernel = tf.get_variable('conv4_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            vgg_W = vgg_weights['conv4_2_W']
            vgg_B = vgg_weights['conv4_2_b']
            kernel = tf.get_variable('conv4_2/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_2/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            vgg_W = vgg_weights['conv4_3_W']
            vgg_B = vgg_weights['conv4_3_b']
            kernel = tf.get_variable('conv4_3/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_3/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            vgg_W = vgg_weights['conv5_1_W']
            vgg_B = vgg_weights['conv5_1_b']
            kernel = tf.get_variable('conv5_1/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_1/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)


        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            vgg_W = vgg_weights['conv5_2_W']
            vgg_B = vgg_weights['conv5_2_b']
            kernel = tf.get_variable('conv5_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)


        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            vgg_W = vgg_weights['conv5_3_W']
            vgg_B = vgg_weights['conv5_3_b']
            kernel = tf.get_variable('conv5_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)


        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')
        print("pool5.shape: %s" % pool5.get_shape())

    return pool5, assign_ops

def load_weights(weight_file):
  weights = np.load(weight_file)
  keys = sorted(weights.keys())
  return weights, keys

class FCNet(object):
    def __init__(self, n_classes = 10):
        self.n_classes = n_classes

    def inference(self, x):
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            # print ("x", x.get_shape())
            with tf.variable_scope("flatten"):
                flatten = tf.reshape(x, [-1, 512])
                # flatten=tf.contrib.layers.flatten(x)
                tf.histogram_summary(tf.get_variable_scope().name+"/layer",flatten)

            with tf.variable_scope("fc1"):
                # kernel=tf.get_variable("w",[flatten.get_shape()[1],384],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[flatten.get_shape()[1],384],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[384],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(flatten,kernel),bias),name='activation')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("fc2"):
                # kernel=tf.get_variable("w",[384,192],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[384,192],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[192],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(layer,kernel),bias),name='activation')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("fc3"):
                # kernel=tf.get_variable("w",[192,self.n_classes],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[192,self.n_classes],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[self.n_classes],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.add(tf.matmul(layer,kernel),bias)
                tf.histogram_summary(tf.get_variable_scope().name+'/softmax',layer)
            
            logits=layer
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits


    def _variable_summaries(self,var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)


    def accuracy(self, logits, labels):
        with tf.name_scope("accuracyVGG"):
            correct_predictions=tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
            accuracy=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.scalar_summary('accuracy',accuracy)

        return accuracy

    def loss(self, logits, labels):
        with tf.name_scope("lossVGG"):
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
            loss = tf.reduce_mean(cross_entropy)
            loss=tf.add(loss, sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
            tf.scalar_summary('loss_regularized',loss)
        return loss
