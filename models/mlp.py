"""
This module implements a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers


class MLP(object):
  """
  This class implements a Multilayer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference and evaluation.
  """

  def __init__(self, n_hidden=[100], n_classes=10, is_training=True,
               activation_fn=tf.nn.relu, dropout_rate=0.,
               weight_initializer=initializers.xavier_initializer(),
               weight_regularizer=regularizers.l2_regularizer(0.001)):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      is_training: Bool Tensor, it indicates whether the model is in training
                        mode or not. This will be relevant for methods that perform
                        differently during training and testing (such as dropout).
                        Have look at how to use conditionals in TensorFlow with
                        tf.cond.
      activation_fn: callable, takes a Tensor and returns a transformed tensor.
                          Activation function specifies which type of non-linearity
                          to use in every hidden layer.
      dropout_rate: float in range [0,1], presents the fraction of hidden units
                         that are randomly dropped for regularization.
      weight_initializer: callable, a weight initializer that generates tensors
                               of a chosen distribution.
      weight_regularizer: callable, returns a scalar regularization loss given
                               a weight variable. The returned loss will be added to
                               the total loss for training purposes.
    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.is_training = is_training
    self.activation_fn = activation_fn
    self.dropout_rate = dropout_rate
    self.weight_initializer = weight_initializer
    self.weight_regularizer = weight_regularizer

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    In order to keep things uncluttered we recommend you (though it's not required)
    to implement a separate function that is used to define a fully connected
    layer of the MLP.

    In order to make your code more structured you can use variable scopes and name
    scopes. You can define a name scope for the whole model, for each hidden
    layer and for output. Variable scopes are an essential component in TensorFlow
    design for parameter sharing. This will be essential in the future when we go
    to recurrent neural networks (RNNs).

    You can use tf.histogram_summary to save summaries of the fully connected layer weights,
    biases, pre-activations, post-activations, and dropped-out activations
    for each layer. It is very useful for introspection of the network using TensorBoard.

    Args:
      x: 2D float Tensor of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """
    # print ("w reg ",self.weight_regularizer)
    # print ("w init",self.weight_initializer)
    with tf.name_scope("nn"):
      layer=x
      for i in xrange(0, len(self.n_hidden)):
        name="hidden"+str(i)
        with tf.variable_scope(name):
          b=tf.get_variable(name+"/b", (self.n_hidden[i],), initializer=tf.constant_initializer(0.1))
          W=tf.get_variable(name+"/w", (layer.get_shape()[1], self.n_hidden[i]), regularizer=self.weight_regularizer, initializer=self.weight_initializer)
          self.variable_summaries(b,name+'/bias')
          self.variable_summaries(W,name+'/weights')

          layer=tf.add(tf.matmul(layer, W), b)
          layer=self.activation_fn(layer)
          tf.histogram_summary(name+'layer',layer)

          layer = tf.cond(self.is_training, lambda:tf.nn.dropout(layer, (1-self.dropout_rate)), lambda: layer)
      name="out"
      with tf.variable_scope(name):
        b=tf.get_variable(name+"/b", (self.n_classes,),initializer=tf.constant_initializer(0.1))
        W=tf.get_variable(name+"/w", (layer.get_shape()[1], self.n_classes),regularizer=self.weight_regularizer, initializer=self.weight_initializer)
        self.variable_summaries(b,'out/bias')
        self.variable_summaries(W,'out/weights')

        layer=tf.add(tf.matmul(layer, W), b)
        tf.histogram_summary('out/layer',layer)

        logits=layer
    return logits
#taken with some changes from https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
  def variable_summaries(self,var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('stddev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.

    You can use tf.scalar_summary to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   observation in batch.
    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """
    with tf.name_scope("loss"):
      loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels),sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
      tf.scalar_summary('loss_regularized',loss)

    return loss

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.

    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each observation in batch.
    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """
    with tf.name_scope("accuracy"):
      correct_predictions=tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
      accuracy=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
      tf.scalar_summary('accuracy',accuracy)

    return accuracy