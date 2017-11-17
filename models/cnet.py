import tensorflow as tf
from collections import deque
import numpy as np
import pdb


class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, kernels=[10, 10], strides=[2], fcUnits=[2], poolingLayerFlag=False,
                 weightInitializer=[tf.contrib.layers.xavier_initializer], activationFunctions=[tf.nn.relu],
                 weightRegularizers=[tf.contrib.layers.l2_regularizer], architecture=['c', 'f', 'd'],
                 regularizationStrength=0.001):

        self.regularizationStrength = regularizationStrength
        self.fcUnits = deque(fcUnits)
        self.strides = deque(strides)
        self.kernels = deque(kernels)
        self.poolingFlag = poolingLayerFlag
        self.functionDict = {}
        self.functionDict['init'] = weightInitializer
        self.functionDict['reg'] = weightRegularizers
        self.functionDict['act'] = activationFunctions

    def inference(self, x):
        '''
        :param x: expected shape (depth , signalSize)
        :return: [alpha, sigma]
        '''
        channels = x.shape[3].value
        with tf.variable_scope('ConvNet'):
            with tf.variable_scope('convLayer'):
                kernelSize = self.kernels.popleft()
                layer = self._depthWiseConvLayer(x, kernelSize, nbChannels=channels, depth=kernelSize,
                                                 activationFunc=self._getFunction('act', 'c'),
                                                 initializer=self._getFunction('init', 'c')())
                maxPooling = self._maxPoolLayer(layer, self.kernels.popleft(), self.strides.popleft())
                # layer = tf.cond(self.poolingFlag, lambda: maxPooling, lambda: layer)
                tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)


            with tf.variable_scope("flatten"):
                # layer = tf.contrib.layers.flatten(layer)
                inShape = layer.get_shape().as_list()
                layer = tf.reshape(layer, [-1, inShape[1] * inShape[2] * inShape[3]])
                tf.summary.histogram(tf.get_variable_scope().name + "/layer", layer)

            with tf.variable_scope('denseLayer'):
                layer = self._denseLayer(layer, self.fcUnits.popleft(), activationFunc=self._getFunction('act', 'd'),
                                         regularizer=self._getFunction('reg', 'd'),
                                         initializer=self._getFunction('init', 'd')())
                tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)

        return layer

    def _depthWiseConvLayer(self, x, kernelSize, depth, nbChannels=1, activationFunc=tf.nn.relu,
                            initializer=tf.contrib.layers.xavier_initializer()):
        '''
        x : input
        kernelSize : int or tuple of kernel dimensions
        depth : number of signal to be convolved
        nbChannels : number of signal channels
        activationFunc : activation
        initializer : weight initializer function
        '''
        # kernel [filter_height, filter_width, in_channels, channel_multiplier]
        kernel = tf.get_variable("w", [1, kernelSize, nbChannels, depth], initializer=initializer)
        bias = tf.get_variable("b", depth * nbChannels, initializer=tf.constant_initializer(0))
        layer = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(layer, bias)
        layer = activationFunc(features=pre_activation, name='activation')

        self._variable_summaries(bias, tf.get_variable_scope().name + '/bias')
        self._variable_summaries(kernel, tf.get_variable_scope().name + '/weights')
        return layer

    def _maxPoolLayer(self, x, kernelSize, stride):
        return tf.nn.max_pool(x, ksize=[1, 1, kernelSize, 1], strides=[1, 1, stride, 1], padding='SAME')

    def _denseLayer(self, x, units, activationFunc=tf.nn.relu, regularizer=tf.contrib.layers.l2_regularizer,
                    initializer=tf.contrib.layers.xavier_initializer()):
        # pdb.set_trace()
        weights = tf.get_variable("w", [x.get_shape()[1], units], regularizer=regularizer(self.regularizationStrength),
                                  initializer=initializer)

        bias = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.1))
        layer = activationFunc(tf.add(tf.matmul(x, weights), bias), name="activation")

        self._variable_summaries(bias, tf.get_variable_scope().name + '/bias')
        self._variable_summaries(weights, tf.get_variable_scope().name + '/weights')
        return layer

    def _getFunction(self, functionType, layerType='c'):
        '''
        :param functionType: act -> activation, init -> initializer, reg -> regularizer
        :param layerType: c-> conv, d -> dense
        :return: activationFunction
        '''
        assert (functionType in self.functionDict), "Not a valid function type (init, reg, act)"
        functionList = self.functionDict[functionType]
        if (type(functionList) == dict):  # dictionary defining functions of different layer types
            if (layerType in functionList):
                return functionList[layerType]
        else:
            if len(functionList) > 0:
                functionList = deque(functionList)
                return functionList.popleft()

        if (functionType == 'init'):
            return tf.contrib.layers.xavier_initializer
        elif (functionType == 'reg'):
            return tf.contrib.layers.l2_regularizer
        elif (functionType == 'act'):
            return tf.nn.relu

    def _variable_summaries(self, var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

            # def accuracy(self, logits, labels):
            #     with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            # tf.summary.scalar('accuracy', accuracy)

            # return accuracy

    def loss(self, pred, y):
        with tf.name_scope("loss"):
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
            # loss = tf.reduce_mean(cross_entropy)
            # loss = tf.add(loss, sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
            loss = tf.losses.mean_squared_error(y, pred)
            tf.summary.scalar('loss_regularized', loss)

        return loss
