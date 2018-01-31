import tensorflow as tf
import numpy as np
from models.neuralNet import *
import pdb, time


class GraphHandler(object):
    def __init__(self, modelPath, modelType, sessConfig, chainedPrefix=""):
        if (modelType.lower() == 'cnn'):
            self.modelType = NeuralNet
        else:
            self.modelType = None
        self.modelPath = modelPath
        self.model = None
        self.session = None
        self.graph = tf.Graph()
        self.prefix = chainedPrefix
        self.predictOperation = None
        self.loss = None
        self.inputPlaceholder = None
        self.outputPlaceholder = None
        self.chainedPlaceholder = None
        self.chainedDict = None
        self.gradientOp = None
        self.sessionConfig = sessConfig
        self.fileName = None
        self.saver = None

    def setSession(self):
        if (self.session is None):
            self.session = tf.Session(graph=self.graph, config=self.sessionConfig)

    def importSavedNN(self, fileName=None, gradientFlag=False):
        self.setSession()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.modelPath + fileName + ".meta", clear_devices=False)
            graph = tf.get_default_graph()
            check = tf.train.get_checkpoint_state(self.modelPath)
            x_pl = graph.get_tensor_by_name("x_pl:0")
            y_pl = graph.get_tensor_by_name("y_pl:0")
            if (self.prefix != ""):
                self.chainedPlaceholder = graph.get_tensor_by_name(self.prefix + "x_pl:0")
            # self.session.run(tf.global_variables_initializer())
            if (fileName is None):
                try:
                    self.saver.restore(self.session, check.model_checkpoint_path)
                except:
                    self.saver.restore(self.session, self.modelPath + fileName)
            else:
                self.fileName = fileName
                self.saver.restore(self.session, self.modelPath + fileName)
            self.predictOperation = tf.get_collection("predict")[0]
            self.inputPlaceholder = x_pl
            self.outputPlaceholder = y_pl
            self.session.run(tf.global_variables_initializer())
            if (gradientFlag):
                self.gradientOp = tf.gradients(self.predictOperation, self.inputPlaceholder)
                # self.gradientOp = tf.gradients(loss, [self.inputPlaceholder, self.outputPlaceholder])

    def buildModel(self, optionDict, chained=None, outPipeline=None, inPipeline=None):
        with self.graph.as_default():
            self.chainedDict = chained
            self.model = self.modelType(volChannels=optionDict['conv_vol_depth'],
                                        irChannels=optionDict['conv_ir_depth'],
                                        pipeline=outPipeline, inPipeline=inPipeline)
            self.loss = self.model.loss(self.predictOperation, self.outputPlaceholder)
            self._delegateModelParams()

    def _delegateModelParams(self):
        if (self.chainedDict is not None):
            self.chainedDict['placeholder'] = self.chainedPlaceholder
        op = self.predictOperation if self.gradientOp is None else self.gradientOp
        self.model.outOp = op
        self.model.derive = True if self.gradientOp is not None else False
        self.model.setChainedDict(self.chainedDict)

    def run(self, data, op=None):
        self.setSession()
        if (op is None):
            if (self.gradientOp is None):
                op = self.predictOperation
            else:
                op = self.gradientOp
        if (self.chainedPlaceholder is not None):
            if (type(data) == tuple):
                inPut = data[0]
                chainedInPut = data[1]
            else:
                raise Exception("No input for the chained model")
            out = self.session.run(op, feed_dict={self.inputPlaceholder: inPut, self.chainedPlaceholder: chainedInPut})
        else:
            if (len(op) > 1):
                if (type(data) == tuple):
                    inPut = data[0]
                    outPut = data[1]
                out = self.session.run(op, feed_dict={self.inputPlaceholder: input, self.outputPlaceholder: outPut})
            else:
                out = self.session.run(op, feed_dict={self.inputPlaceholder: data})
        return out

    def simpleRetrain(self, dataHandler, learningRate, optimization, optionDict, modelName):
        with self.session as sess:
            outPipeline = self.model.getCurrentPipeline()
            inputPipeline = self.model.getCurrentInputPipeline()
            dataHandler.initializePipelines(outPipeline=outPipeline, inputPipeline=inputPipeline)
            testX, testY = dataHandler.getTestData()
            checkpointFolder = optionDict['checkpoint_dir'] + modelName + "/"
            optOperation = optimization
            ttS = time.time()
            max_steps = optionDict['max_steps'] + 1
            epoch = 0
            while epoch < max_steps:
                pdb.set_trace()
                ttS = time.time() if epoch % optionDict['print_frequency'] == 1 else ttS
                batch_x, batch_y = dataHandler.getNextBatch(randomDraw=False)
                _, out = sess.run([optOperation, self.loss], feed_dict={self.inputPlaceholder: batch_x,
                                                                                self.outputPlaceholder: batch_y})
                if epoch % optionDict['print_frequency'] == 0:
                    elapsedTime = time.time() - ttS
                    lrPrint = learningRate if type(learningRate) is float else learningRate.eval()
                    print("====================================================================================")
                    print("Epoch:", '%06d' % (epoch), "Learning rate", '%06f' % (lrPrint), "loss=",
                          "{:.8f}".format(out), "Elapsed time: %03f" % elapsedTime)
                if epoch % optionDict['test_frequency'] == 0 and epoch > 0:
                    if (epoch == optionDict["test_frequency"]):
                        if (outPipeline is not None):
                            print("Transforming testY")
                            testY = outPipeline.transform(testY)
                        if (inputPipeline is not None):
                            print("Transforming testX")
                            for i in range(np.asarray(testX).shape[3]):
                                testX[:, 0, :, i] = inputPipeline.transform(testX[:, 0, :, i])
                    out = sess.run([self.loss], feed_dict={self.inputPlaceholder: testX,
                                                                   self.outputPlaceholder: testY})
                    print("Test set:" "loss=", "{:.10f}".format(out))
                if epoch % optionDict['checkpoint_freq'] == 0 and epoch > 0:
                    if (self.fileName is not None):
                        realEpochNo = int(self.fileName) + epoch
                    else:
                        realEpochNo = epoch
                    self.saver.save(sess, checkpointFolder + str(realEpochNo))
                epoch += 1

    def predict(self, vol, ir, *args):
        self.setSession()
        result = self.model.predict(vol, ir, self.session, self.inputPlaceholder, args)
        return result
