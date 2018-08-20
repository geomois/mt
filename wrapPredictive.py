import argparse, os, time, sys, pdb
import tensorflow as tf
from graphHandler import GraphHandler
from models.neuralNet import NeuralNet
import models.IRCurve as irc
from utils.ahUtils import *
from utils.dataHandler import *
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import utils.customUtils as cu
from utils.FunctionTransformer import FunctionTransformerWithInverse

# region NNDefaults
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 50
MAX_STEPS_DEFAULT = 600
DROPOUT_RATE_DEFAULT = 0.2
TEST_FREQUENCY_DEFAULT = 300
CHECKPOINT_FREQ_DEFAULT = 500
LSTM_UNITS_DEFAULT = '44'
BATCH_WIDTH_DEFAULT = 30
CONVOLUTION_VOL_DEPTH_DEFAULT = 156
CONVOLUTION_IR_DEPTH_DEFAULT = 44
KERNEL_SIZE_DEFAULT = ['10', '10']
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
NUMBER_OF_NODES = ['2']
DEFAULT_ARCHITECTURE = ['c', 'f', 'd']
GPU_MEMORY_FRACTION = 0.3
# endregion NNDefaults

# region QLDefaultConstants
CURRENCY_DEFAULT = 'EUR'
IR_DEFAULT = 'EURIBOR'
# endregion QLDefaultConstants

# region defaultDirectories
DATA_DIR_DEFAULT = 'data/data.h5'
VOL_DATA_DIR_DEFAULT = None  # 'data/_vol.npy'
IR_DATA_DIR_DEFAULT = None  # 'data/_ir.npy'
PARAMS_DATA_DIR_DEFAULT = 'data/_params.npy'
# Directory for tensorflow
LOG_DIR_DEFAULT = 'tf/logs/'
CHECKPOINT_DIR_DEFAULT = 'tf/checkpoints/'
# endregion defaultDirectories

# region self.optionDictionaries
SCALER_DICT = {'standard': StandardScaler,
               'minmax': MinMaxScaler}

WEIGHT_INITIALIZATION_DICT = {'xavier': tf.contrib.layers.xavier_initializer,  # Xavier initialisation
                              'normal': tf.random_normal_initializer,  # Initialization from a standard normal
                              'uniform': tf.random_normal_initializer}  # Initialization from a uniform distribution

WEIGHT_REGULARIZER_DICT = {'none': tf.contrib.layers.l1_regularizer,  # No regularization
                           'l1': tf.contrib.layers.l1_regularizer,  # L1 regularization
                           'l2': tf.contrib.layers.l2_regularizer}  # L2 regularization

ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.tanh,  # Tanh
                   'sigmoid': tf.sigmoid,  # Sigmoid
                   'none': None}

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,  # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer}  # RMSprop

IR_MODEL = {'hullwhite': hullwhite_analytic,
            'g2': g2,
            'g2_local': g2_local}
# endregion self.optionDictionaries

class PSeriesPredict():
    graphHandler = None
    OPTIONS = None
    optionDict = None
    modelName = None
    optionDict = None
    tf.set_random_seed(42)
    np.random.seed(42)

    def buildNN(self, dataHandler):
        if (self.optionDict['architecture'][0] == 'd' or 'l' in self.optionDict['architecture'][0] or 'g' in
                self.optionDict['architecture'][0]):
            if (self.optionDict['architecture'][0] == 'd'):
                mode = 'l'
            else:
                mode = 'p'
            dataHandler.forceSimplify(mode)
        testX, testY = dataHandler.getTestData()

        print("Input dims" + str(testX.shape))
        if (int(self.optionDict['nodes'][len(self.optionDict['nodes']) - 1]) != testY.shape[1]):
            print("LAST fcNodes DIFFERENT SHAPE FROM DATA TARGET")
            print("Aligning...")
            self.optionDict['nodes'][len(self.optionDict['nodes']) - 1] = testY.shape[1]
        print("Output dims" + str((None, testY.shape[1])))
        xShape = None
        if len(testX.shape) == 2:
            xShape = (None, testX.shape[1])
        elif (len(testX.shape) == 3):
            xShape = (None, testX.shape[1], testX.shape[2])
        elif (len(testX.shape) == 4):
            xShape = (None, 1, testX.shape[2], testX.shape[3])

        x_pl = tf.placeholder(tf.float32, shape=xShape, name="x_pl")
        y_pl = tf.placeholder(tf.float32, shape=(None, testY.shape[1]), name="y_pl")
        poolingFlag = tf.placeholder(tf.bool)
        pipeline = None
        outPP, inPP = self.getPipelines(self.optionDict)
        try:
            activation = [ACTIVATION_DICT[act] for act in self.optionDict['activation']]
        except:
            activation = [ACTIVATION_DICT[ACTIVATION_DEFAULT]]
            pass

        nn = NeuralNet(volChannels=self.optionDict['conv_vol_depth'], irChannels=self.optionDict['conv_ir_depth'],
                       kernels=self.optionDict["kernels"], poolingLayerFlag=poolingFlag, inPipeline=inPP, pipeline=outPP,
                       architecture=self.optionDict['architecture'], units=self.optionDict['nodes'],
                       activationFunctions=activation)

        pred = nn.inference(x_pl)
        tf.add_to_collection("predict", pred)
        loss = nn.loss(pred, y_pl)
        tf.add_to_collection("loss", loss)

        return dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline, nn

    def getOptimizerOperation(self, loss, learningRate, global_step):
        optimizer = OPTIMIZER_DICT[self.optionDict['optimizer']](learning_rate=learningRate)
        opt = optimizer.minimize(loss, global_step=global_step)
        return opt

    def getLearningrate(self, decay_rate, global_step):
        if (decay_rate > 0):
            learningRate = tf.train.exponential_decay(learning_rate=self.optionDict['learning_rate'],
                                                      global_step=global_step,
                                                      decay_steps=self.optionDict['decay_steps'],
                                                      decay_rate=self.optionDict['decay_rate'],
                                                      staircase=self.optionDict['decay_staircase'])
        else:
            learningRate = self.optionDict['learning_rate']

        return learningRate

    def trainNN(self, dataHandler, network, loss, pred, x_pl, y_pl, testX, testY):
        mergedSummaries = tf.summary.merge_all()
        saver = tf.train.Saver()
        inputPipeline = outPipeline = None
        try:
            global_step = tf.Variable(0, trainable=False)
            prevTestLoss = 1
            checkpointFolder = self.optionDict['checkpoint_dir'] + self.modelName + "/"
            timestamp = self.modelName + ''.join(str(dt.datetime.now().timestamp()).split('.'))

            learningRate = self.getLearningrate(self.optionDict['decay_rate'], global_step)
            opt = self.getOptimizerOperation(loss, learningRate, global_step)
            if (not self.optionDict['perTermScale']):
                inputPipeline, outPipeline = dataHandler.initializePipelines(inputPipeline=network.inputPipeline,
                                                                             outPipeline=network.pipeline)

            with tf.Session(config=self.getTfConfig()) as sess:

                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter(self.optionDict['log_dir'] + '/train' + timestamp, sess.graph)
                test_writer = tf.summary.FileWriter(self.optionDict['log_dir'] + '/test' + timestamp, sess.graph)

                ttS = time.time()
                max_steps = self.optionDict['max_steps'] + 1
                epoch = 0
                # pdb.set_trace()

                while epoch < max_steps:
                    ttS = time.time() if epoch % self.optionDict['print_frequency'] == 1 else ttS
                    batch_x, batch_y = dataHandler.getNextBatch(randomDraw=False)
                    _, out, merged_sum = sess.run([opt, loss, mergedSummaries],
                                                      feed_dict={x_pl: batch_x, y_pl: batch_y})
                    if epoch % self.optionDict['print_frequency'] == 0:
                        elapsedTime = time.time() - ttS
                        train_writer.add_summary(merged_sum, epoch)
                        train_writer.flush()
                        lrPrint = learningRate if type(learningRate) is float else learningRate.eval()
                        print("====================================================================================")
                        print("Epoch:", '%06d' % (epoch), "Learning rate", '%06f' % (lrPrint), "loss=",
                              "{:.8f}".format(out), "Elapsed time: %03f" % elapsedTime)
                    if epoch % self.optionDict['test_frequency'] == 0 and epoch > 0:
                        if (epoch == self.OPTIONS.test_frequency):
                            if (outPipeline is not None):
                                print("Transforming testY")
                                testY = outPipeline.transform(testY)
                            if (inputPipeline is not None):
                                print("Transforming testX")
                                if (len(testX.shape) == 3):
                                    for j in range(np.asarray(testX).shape[2]):
                                        testX[:, :, j] = inputPipeline.transform(testX[:, :, j])

                                elif (len(testX.shape) == 4):
                                    for j in range(np.asarray(testX).shape[3]):
                                        testX[:, 0, :, j] = inputPipeline.transform(testX[:, 0, :, j])

                        out, merged_sum = sess.run([loss, mergedSummaries], feed_dict={x_pl: testX, y_pl: testY})

                        if (out + 0.00001 <= prevTestLoss and self.optionDict['extend_training']):
                            prevTestLoss = out
                            if (max_steps - epoch <= self.optionDict['test_frequency']):
                                max_steps += self.optionDict['test_frequency'] + 1
                        test_writer.add_summary(merged_sum, epoch)
                        test_writer.flush()
                        print("Test set:" "loss=", "{:.10f}".format(out))
                    if epoch % self.optionDict['checkpoint_freq'] == 0 and epoch > 0:
                        saver.save(sess, checkpointFolder + str(epoch))
                    epoch += 1

            self.optionDict['input_pipeline'] = self.savePipeline(inputPipeline, 'in')
            self.optionDict['pipeline'] = self.savePipeline(outPipeline, 'out')
            opts = self.get_options(False, modelDir=checkpointFolder + str(self.optionDict['max_steps']))

            xShape = None
            if len(testX.shape) == 2:
                xShape = (None, testX.shape[1])
            elif (len(testX.shape) == 3):
                xShape = (None, testX.shape[1], testX.shape[2])
            elif (len(testX.shape) == 4):
                xShape = (None, 1, testX.shape[2], testX.shape[3])
            opts["input_dims"] = xShape
            opts["output_dims"] = (None, testY.shape[1])
            cu.save_obj(opts, checkpointFolder + 'options.pkl')
            tf.reset_default_graph()

        except Exception as e:
            # pdb.set_trace()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print("Exception during training: ", str(e))
            tf.reset_default_graph()

    def setupDataHandler(self, options, allowPredictiveTransformation=True, testPercentage=DEFAULT_TEST_DATA_PERCENTAGE):
        if (type(options) == argparse.Namespace):
            options = vars(options)
        if (options['predictiveShape'] is not None):
            if (options['volFileName'] is not None):
                mode = 'vol'
            elif (options['irFileName'] is not None):
                mode = 'ir'
            else:
                raise ValueError('File name is not correct')

            predictiveShape = (
            mode, options['predictiveShape'], options['channel_range'], allowPredictiveTransformation)
            specialFilePrefix = "_M" + mode + ''.join(options['predictiveShape'])
        else:
            predictiveShape = None
            specialFilePrefix = ''
        if ('testDataPercentage' not in options):
            options['testDataPercentage'] = testPercentage
        dataFileName = options['volFileName'] if options['volFileName'] is not None else options['irFileName']
        dataHandler = DataHandler(dataFileName=dataFileName, batchSize=options['batch_size'],
                                  width=options['batch_width'],
                                  testDataPercentage=options['testDataPercentage'],
                                  volDepth=int(options['data_vol_depth']), irDepth=int(options['data_ir_depth']),
                                  useDataPointers=False, save=options['saveProcessedData'],
                                  specialFilePrefix=specialFilePrefix, predictiveShape=predictiveShape,
                                  targetDataMode=options['target'], perTermScale=options['perTermScale'])
        if (options['processedData']):
            fileList = dataHandler.findTwinFiles(dataFileName)
            simplify = False
            if (options['architecture'][0] == 'd' or 'l' in options['architecture'][0] or 'g' in
                    options['architecture'][0]):
                simplify = True
            dataHandler.delegateDataDictsFromFile(fileList, simplify)

        return dataHandler

    def get_options(self, printFlag=True, modelDir=None):
        options = self.optionDict
        for key, value in self.optionDict.items():
            if (key == 'model_dir' and modelDir is not None):
                value = modelDir
                options[key] = modelDir
            item = key + ' : ' + str(value)
            # array.append((key, value))
            if (printFlag):
                print(item)
        return options

    def initDirectories(self):
        # Make directories if they do not exists yet
        if not tf.gfile.Exists(self.optionDict['log_dir']):
            tf.gfile.MakeDirs(self.optionDict['log_dir'])
        if not tf.gfile.Exists(self.optionDict['data_dir']):
            tf.gfile.MakeDirs(self.optionDict['data_dir'])
        if not tf.gfile.Exists(self.optionDict['checkpoint_dir']):
            tf.gfile.MakeDirs(self.optionDict['checkpoint_dir'])

    def getIrModel(self):
        if (str(self.optionDict['model']) in IR_MODEL):
            return IR_MODEL[self.optionDict['model']]
        else:
            raise RuntimeError("Unknown IR model")

    def getPipelines(self, options):
        inPP = outPP = None
        if (options['use_pipeline']):
            try:
                if (options['pipeline'] is not ""):
                    pipelinePath = options['pipeline']
                else:
                    pipelinePath = options['checkpoint_dir'] + self.modelName + "/pipeline.pkl"
                if (not os.path.exists(pipelinePath)):
                    pipelinePath = None
                outPP = self.buildPipeLine(options['transform'], options['scaler'], pipelinePath)
            except:
                pass
        if (options['use_input_pipeline']):
            try:
                if (options['input_pipeline'] is not ""):
                    pipelinePath = options['input_pipeline']
                else:
                    pipelinePath = options['checkpoint_dir'] + self.modelName + "/input_pipeline.pkl"
                    if (not os.path.exists(pipelinePath)):
                        pipelinePath = None
                inPP = self.buildPipeLine(options['in_transform'], options['in_scaler'], pipelinePath)
            except:
                pass
        return outPP, inPP

    def buildPipeLine(self, transform, scaler, fName=None):
        if (fName is None):
            irModel = self.getIrModel()
            if (transform):
                transformFunc = FunctionTransformerWithInverse(func=irModel["transformation"],
                                                               inv_func=irModel["inverse_transformation"])
            else:
                transformFunc = FunctionTransformerWithInverse(func=None, inv_func=None)

            if (scaler.lower() in SCALER_DICT):
                sc = SCALER_DICT[scaler.lower()]()
            else:
                sc = None
            pipeline = Pipeline([('funcTrm', transformFunc), ('scaler', sc)])
        else:
            try:
                pipeline = joblib.load(fName)
            except:
                pipeline = cu.loadSavedScaler(self.optionDict['input_pipeline'])
        return pipeline

    def savePipeline(self, pipeline, mode):
        pipelinePath = ""
        if (pipeline is not None):
            pipelinePath = self.optionDict['checkpoint_dir'] + self.modelName
            if (not os.path.exists(pipelinePath)):
                os.makedirs(pipelinePath)
            if (mode == 'in'):
                pref = 'input_'
            else:
                pref = 'out_'
            pipelinePath = pipelinePath + '/' + pref + "pipeline.pkl"
            joblib.dump(pipeline, pipelinePath)

        return pipelinePath

    def getTfConfig(self, ):
        if (self.optionDict['use_cpu']):
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            gpuMem = tf.GPUOptions(per_process_gpu_memory_fraction=self.optionDict['gpu_memory_fraction'])
            config = tf.ConfigProto(gpu_options=gpuMem)

        return config

    def setupNetwork(self, options, chainedDict=None, gradientFlag=False, prefix="", inMultipleNetsIndex=None):
        if (type(options) == argparse.Namespace):
            options = vars(options)
        fName, directory = cu.splitFileName(options['model_dir'])
        gh = GraphHandler(directory, options['nn_model'], sessConfig=self.getTfConfig(), chainedPrefix=prefix)
        outPipeline, inPipeline = self.getPipelines(options)
        gh.importSavedNN(fName, gradientFlag=gradientFlag)
        gh.buildModel(options, chained=chainedDict, outPipeline=outPipeline, inPipeline=inPipeline,
                      inMultipleNetsIndex=None)
        return gh

    def buildModelName(self, ps, cr, suff, nn, arch, bw, cvd, cid):
        predShape = ''
        if (ps is not None):
            cR = '_'.join(cr) if type(cr) == list else str(cr)
            predShape = str('Ps' + '_'.join(ps) + '_' + cR + '_')
        mN = suff + predShape + nn + "_A" + ''.join(arch) + "_w" + str(bw) + "_v" + str(cvd) + "_ir" + str(
            cid)

        return mN

    def preparePredictiveNetwork(self):
        channelRange = -1
        gh = self.setupNetwork(options=self.optionDict, gradientFlag=False)
        if (self.optionDict['use_input_pipeline'] and self.optionDict['input_pipeline'] is not None):
            pipelineList = cu.loadSavedScaler(self.optionDict['input_pipeline'])
            gh.model.setInputPipelineList(pipelineList)

        if (len(self.optionDict['channel_range']) > 1):
            channelRange = [int(self.optionDict['channel_range'][0]), int(self.optionDict['channel_range'][1])]
        else:
            channelRange = [0, int(self.optionDict['channel_range'][0])]

        return gh, channelRange

    def saveCalibrationResults(self, res, name=""):
        ffolder = self.optionDict['checkpoint_dir'] + self.modelName + "/"
        path = ffolder + name + ".npy"
        dh = DataHandler()
        try:
            if (self.OPTIONS.force_save):
                np.save(self.optionDict['suffix'] + str(dh.getCurrentRunId()) + name + ".npy", res)
            else:
                if (os.path.exists(path)):
                    path = ffolder + str(dh.getCurrentRunId()) + name + ".npy"
                np.save(path, res)
        except:
            np.save(self.optionDict['suffix'] + str(dh.getCurrentRunId()) + name + ".npy", res)

    def runExportOperations(self):
        if self.optionDict['exportForwardRates']:
            exportPath = './exports/' + self.optionDict['suffix'] + self.optionDict['forwardType'] + ".csv"
            if (not os.path.isfile(exportPath)):
                ir = irc.getIRCurves(self.getIrModel(), currency=self.optionDict['currency'],
                                     irType=self.optionDict['irType'],
                                     irFileName=self.optionDict['irFileName'])

                prime = False
                if (self.optionDict['forwardType'].lower() in 'prime'):
                    prime = True

                ir.calcThetaHW(path=exportPath, prime=prime, skip=self.optionDict['skip'])
            else:
                print("File already exists")

        if self.optionDict['exportInstFw']:
            exportPath = './exports/' + self.optionDict['suffix'] + "instaFw.csv"
            if (not os.path.isfile(exportPath)):
                ir = irc.getIRCurves(self.getIrModel(), currency=self.optionDict['currency'], irType=self.optionDict['irType'],
                                     irFileName=self.optionDict['irFileName'])

                ir.getInstForward(path=exportPath, skip=self.optionDict['skip'])
            else:
                print("File already exists")

    def runTraining(self, retrain=False):
        if (self.optionDict['weight_reg_strength'] is None):
            self.optionDict['weight_reg_strength'] = 0.0

        dh = self.setupDataHandler(self.optionDict)
        if self.optionDict['nn_model'] == 'cnn' or self.optionDict['nn_model'] == 'lstm':
            if retrain:
                gh = self.setupNetwork(options=self.optionDict)
                with gh.graph.as_default():
                    global_step = gh.graph.get_tensor_by_name('Variable:0')
                    learningRate = self.getLearningrate(self.optionDict['decay_rate'], global_step)
                    opt = self.getOptimizerOperation(gh.loss, learningRate, global_step)
                gh.simpleRetrain(dh, learningRate, opt, self.optionDict, self.modelName)
            else:
                dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline, nn = self.buildNN(dh)
                outPipeline, inPipeline = self.getPipelines(self.optionDict)
                nn.inputPipeline = inPipeline
                nn.pipeline = outPipeline
                self.trainNN(dataHandler, nn, loss, pred, x_pl, y_pl, testX, testY)
        else:
            raise ValueError("--train_model argument can be lstm or cnn")

    def runPredict(self, data):
        if(self.graphHandler is None):
            self.graphHandler , _ = self.preparePredictiveNetwork()

        assert (self.graphHandler is not None), "Graph not loaded"
        return self.graphHandler.predict(ir = data)

    def loadOptions(self, path):
        file, model = cu.splitFileName(path)
        opts = cu.load_obj(model + '/options.pkl')
        return opts

    def __init__(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--is_train', action='store_true', help='Train a model')
        parser.add_argument('--retrain', action='store_true', help='Train a model')
        parser.add_argument('-lunits', '--lstm_units', type=str, default=LSTM_UNITS_DEFAULT,
                            help='Comma separated list of number of units in each hidden layer')
        parser.add_argument('-bw', '--batch_width', type=int, default=BATCH_WIDTH_DEFAULT,
                            help='Batch width')
        parser.add_argument('-cvd', '--conv_vol_depth', type=int, default=CONVOLUTION_VOL_DEPTH_DEFAULT,
                            help='Comma separated list of number of convolution depth for volatilities in each layer')
        parser.add_argument('-cid', '--conv_ir_depth', type=int, default=CONVOLUTION_IR_DEPTH_DEFAULT,
                            help='Comma separated list of number of data depth for interest rate in each layer')
        parser.add_argument('-dvd', '--data_vol_depth', type=int, default=CONVOLUTION_VOL_DEPTH_DEFAULT,
                            help='Comma separated list of number of convolution depth for volatilities in each layer')
        parser.add_argument('-did', '--data_ir_depth', type=int, default=CONVOLUTION_IR_DEPTH_DEFAULT,
                            help='Comma separated list of number of data depth for interest rate in each layer')
        parser.add_argument('-fc', '--nodes', nargs='+', default=NUMBER_OF_NODES,
                            help='Comma separated list of number of dense layer depth')
        parser.add_argument('-ar', '--architecture', nargs='+', default=DEFAULT_ARCHITECTURE,
                            help='Comma separated list of characters c->convLayer, l->lstm, f->flatten, d->dense')
        parser.add_argument('--kernels', nargs='+', default=KERNEL_SIZE_DEFAULT, help='Kernel size')
        parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                            help='Learning rate')
        parser.add_argument('-ms', '--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                            help='Number of steps to run trainer.')
        parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                            help='Batch size to run trainer.')
        parser.add_argument('-wi', '--weight_init', type=str, default=WEIGHT_INITIALIZATION_DEFAULT,
                            help='Weight initialization type [xavier, normal, uniform].')
        parser.add_argument('-wis', '--weight_init_scale', type=float, default=WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                            help='Weight initialization scale (e.g. std of a Gaussian).')
        parser.add_argument('-wr', '--weight_reg', type=str, default=WEIGHT_REGULARIZER_DEFAULT,
                            help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
        parser.add_argument('-wrs', '--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                            help='Regularizer strcength for weights of fully-connected layers.')
        parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE_DEFAULT, help='Dropout rate.')
        parser.add_argument('--activation', nargs='+', default=ACTIVATION_DEFAULT,
                            help='Activation function [relu, elu, tanh, sigmoid].')
        parser.add_argument('-o', '--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                            help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
        parser.add_argument('-d', '--data_dir', type=str, default=DATA_DIR_DEFAULT,
                            help='Directory for storing input data')
        parser.add_argument('--processedData', action='store_true', help="Signals that the data is ready to use")
        parser.add_argument('--saveProcessedData', action='store_true', help="Save data being used")
        parser.add_argument('-cf', '--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                            help='How frequently tests will be run')
        parser.add_argument('-tf', '--test_frequency', type=int, default=TEST_FREQUENCY_DEFAULT,
                            help='How frequently tests will be run')
        parser.add_argument('--model_dir', type=str, help='Load trained nn')
        parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT, help='Summaries log directory')
        parser.add_argument('-ird', '--irFileName', type=str, default=IR_DATA_DIR_DEFAULT,
                            help='Interest rate data directory')
        parser.add_argument('-pd', '--paramsFileName', type=str, default=PARAMS_DATA_DIR_DEFAULT,
                            help='Interest rate data directory')
        parser.add_argument('-vold', '--volFileName', type=str, default=VOL_DATA_DIR_DEFAULT,
                            help='Volatility directory')
        parser.add_argument('-ir', '--irType', type=str, default=IR_DEFAULT, help='Interest rate type')
        parser.add_argument('-ccy', '--currency', type=str, default=CURRENCY_DEFAULT, help='Swaption currency')
        parser.add_argument('-nn', '--nn_model', type=str, default='cnn',
                            help='Type of model. Possible options: cnn and lstm')
        parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                            help='Checkpoint directory')
        parser.add_argument('-m', '--model', type=str, default='hullwhite', help='Interest rate model')
        parser.add_argument('--plot', action='store_true', help="Plot results")
        parser.add_argument('-hs', '--historyStart', type=str, default=0, help='History start')
        parser.add_argument('-he', '--historyEnd', type=str, default=-1, help='History end')
        parser.add_argument('--print_frequency', type=int, default=50, help='Frequency of epoch printing')
        parser.add_argument('--skip', type=int, default=0,
                            help='Skip n first dates in history comparison')
        parser.add_argument('-pp', '--pipeline', type=str, default="", help='Pipeline path')
        parser.add_argument('-ipp', '--input_pipeline', type=str, default="", help="Pipeline path for input data")
        parser.add_argument('-cpp', '--custom_pipeline', type=str, default="", help='Custom pipeline path')
        parser.add_argument('--scaler', type=str, default='minmax', help='Scaler')
        parser.add_argument('--in_scaler', type=str, default='minmax', help='Scaler')
        parser.add_argument('-ds', '--decay_steps', type=int, default=3000, help='Decay steps')
        parser.add_argument('-dr', '--decay_rate', type=float, default=0.0, help='Decay rate')
        parser.add_argument('--decay_staircase', action='store_true', help='Decay rate')
        parser.add_argument('--gpu_memory_fraction', type=float, default=GPU_MEMORY_FRACTION,
                            help='Percentage of gpu memory to use')
        parser.add_argument('-ps', '--predictiveShape', nargs='+', default=None,
                            help='Comma separated list of numbers for the input and output depth of the nn')
        parser.add_argument('-cr', '--channel_range', nargs='+', default=['44'],
                            help='Comma separated list specifying the channel range from the data')
        parser.add_argument('--with_gradient', action='store_true',
                            help='Computes partial derivative wrt the neural network input')
        parser.add_argument('-up', '--use_pipeline', action='store_true',
                            help='Use of pipeline for pre-processing')
        parser.add_argument('-uip', '--use_input_pipeline', action='store_true',
                            help='Use of pipeline for pre-processing')
        parser.add_argument('--full_test', action='store_true', help='Calibrate history with new starting points++')
        parser.add_argument('--suffix', type=str, default="", help='Custom string identifier for self.modelName')
        parser.add_argument('--target', type=str, default=None, help='Use specific data target')
        parser.add_argument('--exportForwardRates', action='store_true',
                            help='Calculate and save spot rates to forward rates')
        parser.add_argument('-ft', '--forwardType', type=str, default='theta', help='Theta or prime calculation')
        parser.add_argument('-pts', '--perTermScale', action='store_true', help='Scale input per term')
        parser.add_argument('-eif', '--exportInstFw', action='store_true', help='Export instantaneous forward rates')
        parser.add_argument('--use_cpu', action='store_true', help='Use cpu instead of gpu')
        parser.add_argument('--transform', action='store_true', help="Transform data with pipeline")
        parser.add_argument('--in_transform', action='store_true', help="Transform input data with pipeline")
        parser.add_argument('--extend_training', action='store_true', help="Extend training steps")
        parser.add_argument('--load_options', action='store_true', help="Load options from file")
        parser.add_argument('--force_save', action='store_true', help="Saves in parent folder")

        self.OPTIONS, unparsed = parser.parse_known_args(args=args)

        if (self.OPTIONS.load_options and self.OPTIONS.load_multiple == 0):
            try:
                fileName, model_dir = cu.splitFileName(self.OPTIONS.model_dir)
                self.optionDict = cu.load_obj(model_dir + '/options.pkl')
                if (self.OPTIONS.custom_pipeline != ""):
                    self.optionDict['input_pipeline'] = self.OPTIONS.custom_pipeline
                    self.optionDict['use_input_pipeline'] = True
                self.optionDict['model_dir'] = self.OPTIONS.model_dir  # to use specific checkpoint
                self.optionDict['irFileName'] = self.OPTIONS.irFileName if self.OPTIONS.irFileName is not None else self.optionDict[
                    'irFileName']
                # Keep basic operations
                self.optionDict['exportForwardRates'] = self.OPTIONS.exportForwardRates
                self.optionDict['is_train'] = self.OPTIONS.is_train
                self.optionDict['retrain'] = self.OPTIONS.retrain
                # Keep calibration related input
                self.optionDict['currency'] = self.OPTIONS.currency
                self.optionDict['irType'] = self.OPTIONS.irType
                self.optionDict['exportInstFw'] = False
                self.optionDict['kernels'] = self.OPTIONS.kernels

            except Exception as ex:
                raise Exception("Exception loading option from file:" + str(ex))
        else:
            self.optionDict = vars(self.OPTIONS)
        self.self.modelName = self.buildModelName(ps=self.optionDict['predictiveShape'], cr=self.optionDict['channel_range'], suff=self.optionDict['suffix'], nn=self.optionDict['nn_model'],
                                   arch=self.optionDict['architecture'], bw=self.optionDict['batch_width'],
                                   cvd=self.optionDict['conv_vol_depth'], cid=self.optionDict['conv_ir_depth'])

        _ = self.get_options()
        self.initDirectories()
        # tf.app.run()