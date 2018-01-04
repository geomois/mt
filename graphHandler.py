import tensorflow as tf
import numpy as np
from models.cnet import *
import pdb


class GraphHandler(object):
    def __init__(self, modelPath, modelType, sessConfig, chained=""):
        if (modelType.lower() == 'cnn'):
            self.modelType = ConvNet
        else:
            self.modelType = None
        self.modelPath = modelPath
        self.model = None
        self.session = None
        self.graph = tf.Graph()
        self.prefix = chained
        self.predictOperation = None
        self.inputPlaceholder = None
        self.gradientOp = None
        self.sessionConfig = sessConfig

    def setSession(self):
        if (self.session is None):
            self.session = tf.Session(graph=self.graph, config=self.sessionConfig)

    def importSavedNN(self, fileName=None, gradientFlag=False):
        self.setSession()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.modelPath + fileName + ".meta", clear_devices=False)
            graph = tf.get_default_graph()
            check = tf.train.get_checkpoint_state(self.modelPath)
            x_pl = graph.get_tensor_by_name(self.prefix + "x_pl:0")
            self.session.run(tf.global_variables_initializer())
            if (fileName is None):
                saver.restore(self.session, check.model_checkpoint_path)
            else:
                saver.restore(self.session, self.modelPath + fileName)
            self.predictOperation = tf.get_collection("predict")[0]
            self.inputPlaceholder = x_pl

            if (gradientFlag):
                self.gradientOp = tf.gradients(self.predictOperation, self.inputPlaceholder)

    def buildModel(self, optionDict, pipeline=None):
        op = self.predictOperation if self.gradientOp is None else self.gradientOp
        self.model = self.modelType(volChannels=optionDict['conv_vol_depth'], irChannels=optionDict['conv_ir_depth'],
                                    predictOp=op, pipeline=pipeline,
                                    derive=True if self.gradientOp is not None else False)

    def run(self, data, op=None):
        self.setSession()
        if (op is None):
            if (self.gradientOp is None):
                op = self.predictOperation
            else:
                op = self.gradientOp
        out = self.session.run(op, feed_dict={self.inputPlaceholder: data})
        return out

    def predict(self, vol, ir, *args):
        self.setSession()
        self.model.predict(vol, ir, self.session, self.inputPlaceholder, args)
