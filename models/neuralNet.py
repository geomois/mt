import tensorflow as tf
from collections import deque
import numpy as np
import utils.customUtils as cu
import pdb, re


class NeuralNet(object):
    def __init__(self, volChannels, irChannels, kernels=['10', '10'], depths=[10, 1], poolStrides=[2], units=[2],
                 poolingLayerFlag=False, architecture=['c', 'f', 'd'],
                 activationFunctions=[tf.nn.relu], weightInitializer=[tf.contrib.layers.xavier_initializer],
                 weightRegularizer=[tf.contrib.layers.l2_regularizer], calibrationFunc=None,
                 regularizationStrength=0.001, chainedModel=None, predictOp=None, pipeline=None, inPipeline=None,
                 inMultipleNetsIndex=None, derive=False):

        self.regularizationStrength = regularizationStrength
        self.units = deque([int(i) for i in units])
        self.poolStrides = deque(poolStrides)
        kernels = [int(i) for i in kernels]
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
        if (type(inPipeline) == list):
            self.inputPipelineList = inPipeline
            self.inputPipeline = None
        else:
            self.inputPipelineList = []
            self.inputPipeline = inPipeline
        self.calibrationFunc = calibrationFunc
        self.pipelineList = []
        self.chainedModel = chainedModel
        self.chainedChannel = 0 if chainedModel is None else chainedModel['output_dims'][1]
        self.derive = derive
        self.inMultipleNetsIndex = inMultipleNetsIndex

    def inference(self, x, chainedValues=None):
        '''
        :param x: expected shape (depth , signalSize)
        :param chainedValues: values from chained network
        :return: [alpha, sigma]
        '''
        self.inChannels = x.shape[len(x.shape) - 1].value
        nbChannels = self.inChannels
        layer = None
        flatcount = 0
        densecount = 0
        convcount = 0
        lstmcount = 0
        layer = x
        scope = 'ConvNet'
        if (self.architecture[0] == 'd'):
            scope = "DenseNet"
        elif ('l' in self.architecture[0]):
            scope = "RecurrentNet"

        with tf.variable_scope(scope):
            for i, l in enumerate(self.architecture):
                if (l.lower() == 'c' or l.lower() == 's' or l.lower() == "conv"):
                    if (l.lower() == 's'):
                        separable = True
                    else:
                        separable = False
                    with tf.variable_scope('convLayer' + str(convcount)):
                        kernelSize = self.kernels.popleft()
                        depthSize = self.depths.popleft()
                        layer = self._depthWiseConvLayer(layer, kernelSize, nbChannels=nbChannels,
                                                         depth=depthSize, sep=separable,
                                                         activationFunc=self._getFunction('act', 'c'),
                                                         initializer=self._getFunction('init', 'c')())
                        maxPooling = self._maxPoolLayer(layer, self.kernels.popleft(), self.poolStrides.popleft())
                        layer = tf.cond(self.poolingFlag, lambda: maxPooling, lambda: layer)
                        tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)
                    convcount += 1
                elif ('l' in l.lower() or "lstm" in l.lower() or 'g' in l.lower() or "gru" in l.lower()):
                    with tf.variable_scope("recurrent" + str(lstmcount)):
                        ll = re.findall('(\d+)', l)
                        if (len(ll) > 0):
                            num_layers = int(ll[0])
                        else:
                            num_layers = 1
                        ttype = re.findall('l{1}|g{1}', l)[0]
                        if (len(ttype) == 0):
                            ttype = 'l'
                        layer = self._recurrentLayer(layer, layer.get_shape()[-1].value, num_layers, ttype,
                                                     activationFunc=self._getFunction('act', 'l'),
                                                     regularizer=self._getFunction('reg', 'l'),
                                                     initializer=self._getFunction('init', 'l')())
                elif (l.lower() == 'f' or l.lower() == "flatten"):
                    with tf.variable_scope("flatten" + str(flatcount)):
                        inShape = layer.get_shape().as_list()
                        if (len(inShape) == 4):
                            layer = tf.reshape(layer, [-1, inShape[1] * inShape[2] * inShape[3]])
                        elif (len(inShape) == 3):
                            layer = tf.reshape(layer, [-1, inShape[1] * inShape[2]])

                        if (chainedValues is not None):
                            layer = tf.concat([layer, chainedValues], axis=1)
                        tf.summary.histogram(tf.get_variable_scope().name + "/layer", layer)
                    flatcount += 1
                elif (l.lower() == 'd' or l.lower() == "dense"):
                    with tf.variable_scope('denseLayer' + str(densecount)):
                        layer = self._denseLayer(layer, self.units.popleft(),
                                                 activationFunc=self._getFunction('act', 'd'),
                                                 regularizer=self._getFunction('reg', 'd'),
                                                 initializer=self._getFunction('init', 'd')())
                        tf.summary.histogram(tf.get_variable_scope().name + '/layer', layer)
                    densecount += 1
        return layer

    def _depthWiseConvLayer(self, x, kernelSize, depth, nbChannels=1, sep=False, activationFunc=tf.nn.relu,
                            initializer=tf.contrib.layers.xavier_initializer()):
        if (sep):
            kernel = tf.get_variable("w", [1, kernelSize, nbChannels, depth], initializer=initializer)
            bias = tf.get_variable("b", kernelSize, initializer=tf.constant_initializer(0))
            pointKernel = tf.get_variable("pw", [1, 1, nbChannels * depth, kernelSize], initializer=initializer)
            layer = tf.nn.separable_conv2d(x, kernel, pointKernel, strides=[1, 1, 1, 1], padding='SAME')  # NHWC
        else:
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
        weights = tf.get_variable("w", [x.get_shape()[1], units], regularizer=regularizer(self.regularizationStrength),
                                  initializer=initializer)

        bias = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.1))
        layer = tf.add(tf.matmul(x, weights), bias)
        if (activationFunc is not None):
            layer = activationFunc(layer)
        self._variable_summaries(bias, tf.get_variable_scope().name + '/bias')
        self._variable_summaries(weights, tf.get_variable_scope().name + '/weights')
        return layer

    def _recurrentLayer(self, x, units, num_layers, ttype, activationFunc=None,
                        regularizer=tf.contrib.layers.l2_regularizer,
                        initializer=tf.contrib.layers.xavier_initializer()):
        if (ttype == 'l'):
            recurrentModule = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(units) for _ in range(num_layers)])
        else:
            recurrentModule = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(units) for _ in range(num_layers)])

        outputs, current_state = tf.nn.dynamic_rnn(recurrentModule, x, dtype=tf.float32)
        weights = tf.get_variable("w", [units, units], initializer=initializer)
        bias = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.1))
        layer = tf.matmul(current_state[-1].h, weights) + bias
        return layer

    def _getFunction(self, functionType, layerType='c'):
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

    def loss(self, pred, y):
        with tf.name_scope("loss"):
            loss = tf.losses.mean_squared_error(y, pred)
            tf.summary.scalar('loss_regularized', loss)

        return loss

    def setChainedDict(self, chainedModel):
        self.chainedModel = chainedModel
        self.chainedChannel = 0 if chainedModel is None else chainedModel['output_dims'][1]

    def setPipelineList(self, plist):
        self.pipelineList = plist

    def setInputPipelineList(self, plist):
        if (type(plist) == list):
            self.inputPipelineList = plist
            self.inputPipeline = None
        else:
            self.inputPipelineList = []
            self.inputPipeline = plist

    def getCurrentPipeline(self, index=None, mapIndex=False):
        if (len(self.pipelineList) > 1 and index is not None):
            if (mapIndex and len(self.pipelineList) > cu.index_Map_Libor_Eonia[-1]):
                index = cu.index_Map_Libor_Eonia[index]
            self.pipeline = self.pipelineList[index]
        return self.pipeline

    def getCurrentInputPipeline(self, index=None, mapIndex=False):
        if (index == None and self.inMultipleNetsIndex is not None):
            index = self.inMultipleNetsIndex
        if (len(self.inputPipelineList) > 1 and index is not None):
            if (mapIndex and len(self.inputPipelineList) == cu.index_Map_Libor_Eonia[-1]):
                index = cu.index_Map_Libor_Eonia[index]
            self.inputPipeline = self.inputPipelineList[index]
        return self.inputPipeline

    @staticmethod
    def _getTransformationFunction(transformationType, step, pp):
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

    @staticmethod
    def npToTfFunc(func, inPut):
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

    def applyPipeLine(self, tType, x, mode, useTf=False, mapIndex=False):
        if (mode == 'input'):
            modeFunc = self.getCurrentInputPipeline
        else:
            modeFunc = self.getCurrentPipeline
        for i in range(x.shape[1]):
            pp = modeFunc(i, mapIndex)
            func = self._getTransformationFunction(tType, 'pre', pp)
            if func is not None:
                if (useTf):
                    x = self.npToTfFunc(func, x)
                else:
                    x[:, i] = func(x[:, i])
            try:
                x[:, i] = self._getTransformationFunction(tType, 'scale', pp)(x[:, i].reshape((-1, 1)))[:, 0]
            except:
                try:
                    x[:, i] = self._getTransformationFunction(tType, 'scale', pp)(x[:, i:i + 1].T)
                except:
                    x[:, i] = self._getTransformationFunction(tType, 'scale', pp)(x.T)
        return x

    def derivationProc(self, out, totalDepth, xShape):
        der = cu.transformDerivatives(out, 0, totalDepth, xShape)
        if (der.shape[1] > 1):
            out = np.average(der, axis=0).reshape((-1, 1))
        else:
            out = np.average(der).reshape((-1, 1))
        return out

    def predict(self, vol, ir, sess, x_pl, *args):
        mapIndex = False
        if (ir.shape[1] != self.irChannels):
            ir = ir[:, cu.index_Map_Libor_Eonia]
            mapIndex = True
        chained_pl = chainedOutput = None
        if (self.chainedModel is not None):
            chainedOutput = self.chainedModel['model'].predict(vol, ir)
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
            x = self.applyPipeLine('transform', x, 'input', useTf=False, mapIndex=mapIndex)
        if (len(x_pl.get_shape()) == 3):
            x = x.reshape((1, x.shape[0], x.shape[1]))
        elif (len(x_pl.get_shape()) == 4):
            x = x.reshape((1, 1, x.shape[0], x.shape[1]))
        if (chainedOutput is not None):
            out = sess.run(self.outOp, feed_dict={x_pl: x, chained_pl: chainedOutput})
        else:
            out = sess.run(self.outOp, feed_dict={x_pl: x})

        if (self.derive):
            if (len(out[0].shape) == 3):
                out[0] = out[0].reshape((1, 1, out[0].shape[1], out[0].shape[2]))
                x = x.reshape((1, 1, x.shape[1], x.shape[2]))
            out = self.derivationProc(out, totalDepth, x.shape)
        else:
            if (len(self.pipelineList) > 0 or self.pipeline is not None):
                pdb.set_trace()
                if (len(self.pipelineList) == 0):
                    out = self.pipeline.inverse_transform(np.asarray(out))
                else:
                    out = self.applyPipeLine('inverse', out, 'output', useTf=False, mapIndex=mapIndex)

            if (chainedOutput is not None):
                out = np.append(chainedOutput, out)
            out = out.reshape((1, -1))
            out = out.tolist()
        return out
