import tensorflow as tf
from collections import deque
import numpy as np
import utils.customUtils as cu
import pdb


class ConvNet(object):
    def __init__(self, volChannels, irChannels, kernels=[10, 10], depths=[10, 1], poolStrides=[2], fcUnits=[2],
                 poolingLayerFlag=False, architecture=['c', 'f', 'd'],
                 activationFunctions=[tf.nn.relu], weightInitializer=[tf.contrib.layers.xavier_initializer],
                 weightRegularizer=[tf.contrib.layers.l2_regularizer], calibrationFunc=None,
                 regularizationStrength=0.001, chainedModel=None, predictOp=None, pipeline=None, inPipeline=None,
                 derive=False):

        self.regularizationStrength = regularizationStrength
        self.fcUnits = deque([int(i) for i in fcUnits])
        self.poolStrides = deque(poolStrides)
        self.kernels = deque(kernels)
        self.depths = deque(depths)
        self.poolingFlag = poolingLayerFlag
        self.functionDict = {'init': weightInitializer, 'reg': weightRegularizer, 'act': activationFunctions}
        self.functionDictDefault = {'init': weightInitializer, 'reg': weightRegularizer, 'act': activationFunctions}
        self.outOp = predictOp
        self.inChannels = volChannels + irChannels
        self.volChannels = volChannels
        self.irChannels = irChannels
        self.inKernelSize = self.kernels[0]
        self.architecture = architecture
        self.pipeline = pipeline
        self.inputPipeline = inPipeline
        self.calibrationFunc = calibrationFunc
        self.pipelineList = []
        self.inputPipelineList = []
        self.chainedModel = chainedModel
        self.chainedChannel = 0 if chainedModel is None else chainedModel['output_dims'][1]
        self.derive = derive

    def inference(self, x, chainedValues=None):
        '''
        :param x: expected shape (depth , signalSize)
        :param chainedValues: values from chained network
        :return: [alpha, sigma]
        '''
        self.inChannels = x.shape[3].value
        nbChannels = self.inChannels
        layer = None
        flatcount = 0
        densecount = 0
        convcount = 0
        layer = x
        with tf.variable_scope('ConvNet'):
            for i, l in enumerate(self.architecture):
                if (l.lower() == 'c' or l.lower() == "conv"):
                    with tf.variable_scope('convLayer' + str(convcount)):
                        kernelSize = self.kernels.popleft()
                        depthSize = self.depths.popleft()
                        # if (layer is not None):
                        #     nbChannels = layer.get_shape().as_list()[3]
                        layer = self._depthWiseConvLayer(layer, kernelSize, nbChannels=nbChannels,
                                                         depth=depthSize,
                                                         activationFunc=self._getFunction('act', 'c'),
                                                         initializer=self._getFunction('init', 'c')())
                        maxPooling = self._maxPoolLayer(layer, self.kernels.popleft(), self.poolStrides.popleft())
                        # layer = tf.cond(self.poolingFlag, lambda: maxPooling, lambda: layer)
                        tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)
                    convcount += 1
                elif (l.lower() == 'f' or l.lower() == "flatten"):
                    with tf.variable_scope("flatten" + str(flatcount)):
                        # layer = tf.contrib.layers.flatten(layer)
                        inShape = layer.get_shape().as_list()
                        layer = tf.reshape(layer, [-1, inShape[1] * inShape[2] * inShape[3]])
                        if (chainedValues is not None):
                            # pdb.set_trace()
                            layer = tf.concat([layer, chainedValues], axis=1)
                        tf.summary.histogram(tf.get_variable_scope().name + "/layer", layer)
                    flatcount += 1
                elif (l.lower() == 'd' or l.lower() == "dense"):
                    with tf.variable_scope('denseLayer' + str(densecount)):
                        layer = self._denseLayer(layer, self.fcUnits.popleft(),
                                                 activationFunc=self._getFunction('act', 'd'),
                                                 regularizer=self._getFunction('reg', 'd'),
                                                 initializer=self._getFunction('init', 'd')())
                        tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)
                    densecount += 1
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
        layer = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')  # NHWC
        pre_activation = tf.nn.bias_add(layer, bias)
        layer = activationFunc(pre_activation, name='activation')

        self._variable_summaries(bias, tf.get_variable_scope().name + '/bias')
        self._variable_summaries(kernel, tf.get_variable_scope().name + '/weights')
        return layer

    def _maxPoolLayer(self, x, kernelSize, stride):
        return tf.nn.max_pool(x, ksize=[1, 1, kernelSize, 1], strides=[1, 1, stride, 1], padding='SAME')

    def _denseLayer(self, x, units, activationFunc=None, regularizer=tf.contrib.layers.l2_regularizer,
                    initializer=tf.contrib.layers.xavier_initializer()):
        # pdb.set_trace()
        weights = tf.get_variable("w", [x.get_shape()[1], units], regularizer=regularizer(self.regularizationStrength),
                                  initializer=initializer)

        bias = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.1))
        layer = tf.add(tf.matmul(x, weights), bias)
        if (activationFunc is not None):
            layer = activationFunc(layer)
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
                self.functionDict[functionType] = deque(functionList)
                return self.functionDict[functionType].popleft()

        return self.functionDictDefault[functionType][0]
        # if (functionType == 'init'):
        #     return tf.contrib.layers.xavier_initializer
        # elif (functionType == 'reg'):
        #     return return self.functionDict[functionType][0]
        # elif (functionType == 'act'):
        #     return self.functionDict[functionType][0]

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
            # tf.losses.log_loss()
            loss = tf.losses.mean_squared_error(y, pred)
            tf.summary.scalar('loss_regularized', loss)

        return loss

    def calibrationLoss(self, date):
        pass

    def setPipelineList(self, plist):
        self.pipelineList = plist

    def setInputPipelineList(self, plist):
        self.inputPipelineList = plist

    def getCurrentPipeline(self, index):
        if (len(self.pipelineList) > 1):
            self.pipeline = self.pipelineList[index]
        return self.pipeline

    def getCurrentInputPipeline(self, index):
        if (len(self.inputPipelineList) > 1):
            self.inputPipeline = self.inputPipelineList[index]
        return self.inputPipeline

    def _getTransformationFunction(self, transformationType, step, pp):
        if (pp.steps is not None):
            if (transformationType.lower() == "transform"):
                if (step.lower() == 'scale'):
                    return pp.steps[1][1].transform
                else:
                    return pp.steps[0][1].func
            else:
                if (step.lower() == 'scale'):
                    return pp.steps[1][1].inverse_transform
                else:
                    return pp.steps[0][1].inv_func
        else:
            raise Exception("No scaler found")

    def npToTfFunc(self, func, inPut):
        if (func is not None):
            if (func is np.exp):
                inPut = tf.exp(inPut)
            elif (func is np.log):
                inPut = tf.log(inPut)
            else:
                inPut = func(inPut)
        else:
            pass

        return inPut

    def applyPipeLine(self, tType, x, mode, useTf=False):
        if (mode == 'input'):
            modeFunc = self.getCurrentInputPipeline
        else:
            modeFunc = self.getCurrentPipeline
        for i in range(x.shape[1]):
            # if (i <= len(self.pipelineList)):
            pp = modeFunc(i)
            pdb.set_trace()
            if(tType == 'inverse'):
                x = pp.inverse_transform(x[:,i])
            else:
                x = pp.inverse_transform(x[:, i])
            # func = self._getTransformationFunction(tType, 'pre', pp)
            # if (useTf):
            #     x = self.npToTfFunc(func, x)
            # else:
            #     x[:, i] = func(x[:, i])
            # x[:, i] = self._getTransformationFunction(tType, 'scale', pp)(x[:, i].reshape((-1, 1)))[:, 0]
        return x

    def derivationProc(self, out, totalDepth, xShape):
        der = cu.transformDerivatives(out, 0, totalDepth, xShape)
        if (der.shape[1] > 1):
            out = np.average(der, axis=0).reshape((-1, 1))
        else:
            out = np.average(der).reshape((-1, 1))
        if(out<0):
            out = np.abs(out)
        # print(der.shape)
        # print(der[len(der) - 1, 0], np.average(der))
        # out = [der[0, 0]]
        # out = [der[len(der) - 1, 0]]

        # out = [0.025]
        # out = [0.001]
        return out

    def predict(self, vol, ir, sess, x_pl, *args):
        chainedOutput = None
        chained_pl = None
        if (self.chainedModel is not None):
            chainedOutput = self.chainedModel['model'].predict(vol, ir)
            # if (len(self.pipelineList) > 0 or self.inputPipeline is not None):
            #     chainedOutput = self.getCurrentInputPipeline(0).transform(chainedOutput)
            chained_pl = self.chainedModel['placeholder']

        totalDepth = self.volChannels + self.irChannels
        x = np.empty((0, totalDepth))
        if (self.volChannels > 0):
            x = np.float32(vol[:, :self.volChannels])
        if (self.irChannels > 0):
            if (x.shape[1] != totalDepth):
                x = np.hstack((x, np.float32(ir[:, :self.irChannels])))
            else:
                x = np.vstack((x, np.float32(ir[:, :self.irChannels])))

        if (len(self.inputPipelineList) > 0 or self.inputPipeline is not None):
            x = self.applyPipeLine('transform', x, 'input', useTf=False)
        x = x.reshape((1, 1, x.shape[0], x.shape[1]))
        if (chainedOutput is not None):
            out = sess.run(self.outOp, feed_dict={x_pl: x, chained_pl: chainedOutput})
        else:
            out = sess.run(self.outOp, feed_dict={x_pl: x})
        if (self.derive):
            out = self.derivationProc(out, totalDepth, x.shape)
        else:
            if (len(self.pipelineList) or self.pipeline is not None):
                # out = self.applyPipeLine('inverse', out, 'output', useTf=False)
                pdb.set_trace()
                out = self.pipeline.inverse_transform(np.asarray(out).reshape((1,2)))
            if (chainedOutput is not None):
                out = np.append(chainedOutput, out)
            out = out.reshape((1, -1))
            out = out.tolist()

        return out
