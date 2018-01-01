import argparse
import os
import tensorflow as tf
import numpy as np
from models.cnet import ConvNet
import models.instruments as inst
from dataUtils.dataHandler import DataHandler
import pdb
import datetime as dt
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import dataUtils.customUtils as cu

# region NNDefaultConstants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 50
MAX_STEPS_DEFAULT = 600
DROPOUT_RATE_DEFAULT = 0.2
TEST_FREQUENCY_DEFAULT = 300
CHECKPOINT_FREQ_DEFAULT = 500
DNN_HIDDEN_UNITS_DEFAULT = '100'
BATCH_WIDTH_DEFAULT = 30
CONVOLUTION_VOL_DEPTH_DEFAULT = 156
CONVOLUTION_IR_DEPTH_DEFAULT = 44
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
FULLY_CONNECTED_NODES_DEFAULT = ['2']
DEFAULT_ARCHITECTURE = ['c', 'f', 'd']
GPU_MEMORY_FRACTION = 0.3
# endregion NNDefaultConstants

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

# region optionDictionaries
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

IR_MODEL = {'hullwhite': inst.hullwhite_analytic,
            'g2': inst.g2,
            'g2_local': inst.g2_local
            }
# endregion optionDictionaries

OPTIONS = None
modelName = None
tf.set_random_seed(42)
np.random.seed(42)


def trainLSTM(dataHandler):
    tf.set_random_seed(42)
    np.random.seed(42)
    if OPTIONS.dnn_hidden_units:
        dnn_hidden_units = OPTIONS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []


def buildCnn(dataHandler, swaptionGen=None):
    testX, testY = dataHandler.getTestData()
    print("Input dims" + str((None, 1, testX.shape[2], testX.shape[3])))
    if (int(OPTIONS.fullyConnectedNodes[len(OPTIONS.fullyConnectedNodes) - 1]) != testY.shape[1]):
        print("LAST fcNodes DIFFERENT SHAPE FROM DATA TARGET")
        print("Aligning...")
        OPTIONS.fullyConnectedNodes[len(OPTIONS.fullyConnectedNodes) - 1] = testY.shape[1]
    print("Output dims" + str((None, testY.shape[1])))
    x_pl = tf.placeholder(tf.float32, shape=(None, 1, testX.shape[2], testX.shape[3]), name="x_pl")
    y_pl = tf.placeholder(tf.float32, shape=(None, testY.shape[1]), name="y_pl")
    poolingFlag = tf.placeholder(tf.bool)
    pipeline = None
    # if (OPTIONS.pipeline is not ""):
    #     # pipeline = dataHandler.fitPipeline(getPipeLine())
    #     pipeline = getPipeLine()

    if (OPTIONS.use_pipeline):
        pipeline = dataHandler.initializePipeline(getPipeLine())
    try:
        activation = [ACTIVATION_DICT[act] for act in OPTIONS.activation]
    except:
        activation = [ACTIVATION_DICT[ACTIVATION_DEFAULT]]
        pass
    cnn = ConvNet(volChannels=OPTIONS.conv_vol_depth, irChannels=OPTIONS.conv_ir_depth, poolingLayerFlag=poolingFlag,
                  architecture=OPTIONS.architecture, fcUnits=OPTIONS.fullyConnectedNodes, pipeline=pipeline,
                  activationFunctions=activation,
                  calibrationFunc=swaptionGen.calibrate if swaptionGen is not None else None)
    pred = cnn.inference(x_pl)
    tf.add_to_collection("predict", pred)
    if (OPTIONS.use_calibration_loss):
        y_pl = tf.placeholder(tf.int32, shape=(None, testY.shape[1]), name="y_pl")
        loss = cnn.calibrationLoss(pred)
    else:
        loss = cnn.loss(pred, y_pl)

    return dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline


def trainNN(dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline=None):
    mergedSummaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    try:
        global_step = tf.Variable(0, trainable=False)
        prevTestLoss = 1
        with tf.Session(config=getTfConfig()) as sess:
            sess.run(tf.global_variables_initializer())
            timestamp = modelName + ''.join(str(dt.datetime.now().timestamp()).split('.'))
            train_writer = tf.summary.FileWriter(OPTIONS.log_dir + '/train' + timestamp, sess.graph)
            test_writer = tf.summary.FileWriter(OPTIONS.log_dir + '/test' + timestamp, sess.graph)

            if (OPTIONS.decay_rate > 0):
                learningRate = tf.train.exponential_decay(learning_rate=OPTIONS.learning_rate,
                                                          global_step=global_step,
                                                          decay_steps=OPTIONS.decay_steps,
                                                          decay_rate=OPTIONS.decay_rate,
                                                          staircase=OPTIONS.decay_staircase)
            else:
                learningRate = OPTIONS.learning_rate

            optimizer = OPTIMIZER_DICT[OPTIONS.optimizer](learning_rate=learningRate)
            opt = optimizer.minimize(loss, global_step=global_step)

            gradient = None
            if (OPTIONS.with_gradient):
                # gradient = tf.gradients(loss, x_pl)
                gradient = tf.gradients(pred, x_pl)

            checkpointFolder = OPTIONS.checkpoint_dir + modelName + "/"
            ttS = 0
            max_steps = OPTIONS.max_steps
            epoch = 0
            while epoch < max_steps:
                ttS = os.times().elapsed if epoch % OPTIONS.print_frequency == 1 else ttS
                batch_x, batch_y = dataHandler.getNextBatch(pipeline=pipeline, randomDraw=False)
                _, out, merged_sum = sess.run([opt, loss, mergedSummaries],
                                              feed_dict={x_pl: batch_x, y_pl: batch_y})

                if epoch % OPTIONS.print_frequency == 0:
                    elapsedTime = os.times().elapsed - ttS
                    train_writer.add_summary(merged_sum, epoch)
                    train_writer.flush()
                    lrPrint = learningRate if type(learningRate) is float else learningRate.eval()
                    print("====================================================================================")
                    print("Epoch:", '%06d' % (epoch), "Learning rate", '%06f' % (lrPrint), "loss=",
                          "{:.6f}".format(out), "Elapsed time: %03f" % elapsedTime)
                if epoch % OPTIONS.test_frequency == 0 and epoch > 0:
                    if (epoch == OPTIONS.test_frequency and pipeline is not None):
                        if (pipeline.steps[1][1] is not None):  # apply scaler
                            print("Transforming test")
                            # testY = pipeline.transform(testY)
                            testX = pipeline.transform(testX)
                    out, merged_sum = sess.run([loss, mergedSummaries], feed_dict={x_pl: testX, y_pl: testY})
                    if (out + 0.00001 < prevTestLoss and OPTIONS.extend_training):
                        if (max_steps - epoch <= OPTIONS.test_frequency):
                            max_steps += OPTIONS.test_frequency + 1
                    test_writer.add_summary(merged_sum, epoch)
                    test_writer.flush()
                    print("Test set:" "loss=", "{:.6f}".format(out))
                if epoch % OPTIONS.checkpoint_freq == 0 and epoch > 0:
                    saver.save(sess, checkpointFolder + str(epoch))
                epoch += 1

            if (gradient is not None):
                derivative = sess.run(gradient, feed_dict={x_pl: testX})
                transformDerivatives(derivative, dataHandler, testX, folder=checkpointFolder)

        tf.reset_default_graph()
    except Exception as ex:
        # pdb.set_trace()
        print("Exception during training: ", str(ex))
        tf.reset_default_graph()

    return pipeline


def transformDerivatives(derivative, dataHandler, testX, folder=None, save=True):
    der = cu.transformDerivatives(derivative, dataHandler.channelStart, dataHandler.channelEnd, testX)
    if (save):
        if (folder is None):
            folder = OPTIONS.checkpoint_dir + modelName + "/"
        np.save(folder + "testDerivatives.npy", der)

    return der


def importSavedNN(session, modelPath, fileName):
    saver = tf.train.import_meta_graph(modelPath + fileName + ".meta", clear_devices=True)
    graph = tf.get_default_graph()
    check = tf.train.get_checkpoint_state(modelPath)
    x_pl = graph.get_tensor_by_name("x_pl:0")
    # x_pl = tf.placeholder(tf.float32, shape=(None, 1, 50, 5))
    session.run(tf.global_variables_initializer())
    # saver.restore(session, check.model_checkpoint_path)
    saver.restore(session, modelPath + fileName)
    return tf.get_collection("predict")[0], x_pl


def setupDataHandler():
    if (OPTIONS.predictiveShape is not None):
        if (OPTIONS.volFileName is not None):
            mode = 'vol'
        elif (OPTIONS.irFileName is not None):
            mode = 'ir'
        else:
            raise ValueError('File name is not correct')
        predShape = (mode, OPTIONS.predictiveShape, OPTIONS.channel_range)
        specialFilePrefix = "_M" + mode + ''.join(OPTIONS.predictiveShape)
    else:
        predShape = None
        specialFilePrefix = ''

    dataFileName = OPTIONS.volFileName if OPTIONS.volFileName is not None else OPTIONS.irFileName
    dataHandler = DataHandler(dataFileName=dataFileName, batchSize=OPTIONS.batch_size, width=OPTIONS.batch_width,
                              volDepth=int(OPTIONS.data_vol_depth), irDepth=int(OPTIONS.data_ir_depth),
                              useDataPointers=OPTIONS.use_calibration_loss, save=OPTIONS.saveProcessedData,
                              specialFilePrefix=specialFilePrefix, predictiveShape=predShape,
                              targetDataMode=OPTIONS.target)
    if (OPTIONS.processedData):
        fileList = dataHandler.findTwinFiles(dataFileName)
        dataHandler.delegateDataDictsFromFile(fileList)

    return dataHandler


def print_flags():
    for key, value in vars(OPTIONS).items():
        print(key + ' : ' + str(value))


def initDirectories():
    # Make directories if they do not exists yet
    if not tf.gfile.Exists(OPTIONS.log_dir):
        tf.gfile.MakeDirs(OPTIONS.log_dir)
    if not tf.gfile.Exists(OPTIONS.data_dir):
        tf.gfile.MakeDirs(OPTIONS.data_dir)
    if not tf.gfile.Exists(OPTIONS.checkpoint_dir):
        tf.gfile.MakeDirs(OPTIONS.checkpoint_dir)


def getIrModel():
    if (str(OPTIONS.model) in IR_MODEL):
        return IR_MODEL[OPTIONS.model]
    else:
        raise RuntimeError("Unknown IR model")


def getPipeLine(fileName=None):
    if (fileName is None):
        irModel = getIrModel()
        if (OPTIONS.no_transform):
            transformFunc = inst.FunctionTransformerWithInverse(func=None, inv_func=None)
        else:
            transformFunc = inst.FunctionTransformerWithInverse(func=irModel["transformation"],
                                                                inv_func=irModel["inverse_transformation"])

        if (OPTIONS.scaler.lower() in SCALER_DICT):
            scaler = SCALER_DICT[OPTIONS.scaler]()
        else:
            scaler = None

        pipeline = Pipeline([('funcTrm', transformFunc), ('scaler', scaler)])
    else:
        pipeline = joblib.load(fileName)

    return pipeline


def getTfConfig():
    if (OPTIONS.use_cpu):
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpuMem = tf.GPUOptions(per_process_gpu_memory_fraction=OPTIONS.gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpuMem)

    return config


def setupNetwork(session, gradientFlag=False):
    fileName = ''.join(re.findall(r'(/)(\w+)', OPTIONS.model_dir).pop())
    directory = OPTIONS.model_dir.split(fileName)[0]
    predictOp, x_pl = importSavedNN(session, directory, fileName)
    gradientOp = None
    operation = predictOp
    if (gradientFlag):
        gradientOp = tf.gradients(predictOp, x_pl)
        operation = gradientOp
    pipeline = None
    if (OPTIONS.use_pipeline):
        try:
            if (OPTIONS.pipeline is not ""):
                pipelinePath = OPTIONS.pipeline
            else:
                pipelinePath = OPTIONS.checkpoint_dir + '/' + modelName + "/pipeline.pkl"
            pipeline = getPipeLine(pipelinePath)
        except:
            pass

    if (OPTIONS.nn_model.lower() == 'cnn'):
        model = ConvNet(volChannels=OPTIONS.conv_vol_depth, irChannels=OPTIONS.conv_ir_depth,
                        predictOp=operation, pipeline=pipeline, derive=gradientFlag)
    elif (OPTIONS.nn_model.lower() == 'lstm'):
        # model = LSTM(predictOp = predictOp)
        pass
    return model, predictOp, x_pl, gradientOp


def savePipeline(pipeline):
    if (pipeline is not None):
        pipelinePath = OPTIONS.checkpoint_dir + modelName
        if (not os.path.exists(pipelinePath)):
            os.makedirs(pipelinePath)
        joblib.dump(pipeline, pipelinePath + "/pipeline.pkl", compress=1)


def main(_):
    print_flags()
    initDirectories()
    swo = None
    if (OPTIONS.weight_reg_strength is None):
        OPTIONS.weight_reg_strength = 0.0
    inst.setDataFileName(OPTIONS.data_dir)
    if (OPTIONS.calibrate or OPTIONS.use_calibration_loss or OPTIONS.exportForwardRates):
        swo = inst.get_swaptiongen(getIrModel(), currency=OPTIONS.currency, irType=OPTIONS.irType,
                                   volFileName=OPTIONS.volFileName, irFileName=OPTIONS.irFileName)
        if OPTIONS.calibrate:
            swo.calibrate_history(start=int(OPTIONS.historyStart), end=int(OPTIONS.historyEnd))

        if OPTIONS.exportForwardRates:
            exportPath = './exports/' + OPTIONS.suffix + 'fwCurves' + "_fDays" + str(OPTIONS.futureIncrement) + ".csv"
            if (not os.path.isfile(exportPath)):
                swo.calcForward(path=exportPath, futureIncrementInDays=OPTIONS.futureIncrement)
            else:
                print("File already exists")

    if OPTIONS.is_train:
        dh = setupDataHandler()
        if OPTIONS.nn_model == 'cnn':
            dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline = buildCnn(dh, swaptionGen=swo)
            pipeline = trainNN(dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline)
            savePipeline(pipeline)
        elif OPTIONS.nn_model == 'lstm':
            trainLSTM(dh)
        else:
            raise ValueError("--train_model argument can be lstm or cnn")

    if OPTIONS.calculate_gradient:
        with tf.Session(config=getTfConfig()) as sess:
            model, predictOp, x_pl, gradient = setupNetwork(sess, gradientFlag=True)
            if OPTIONS.with_gradient:
                dh = setupDataHandler()
                testX, _ = dh.getTestData()
                deriv = sess.run(gradient, feed_dict={x_pl: testX})
                path = OPTIONS.checkpoint_dir if OPTIONS.checkpoint_dir != CHECKPOINT_DIR_DEFAULT else None
                transformDerivatives(deriv, dh, testX, path)
            else:
                if (OPTIONS.use_pipeline and OPTIONS.pipeline is not None):
                    pipelineList = cu.loadSavedScaler(OPTIONS.pipeline)
                    model.setPipelineList(pipelineList)

                swo = inst.get_swaptiongen(getIrModel(), OPTIONS.currency, OPTIONS.irType)
                if (OPTIONS.calibrate_sigma):
                    if (len(OPTIONS.channel_range) > 1):
                        channelRange = [int(OPTIONS.channel_range[0]), int(OPTIONS.channel_range[1])]
                    else:
                        channelRange = [0, int(OPTIONS.channel_range[0])]
                    sigmas = swo.calibrate_sigma(model, modelName, dataLength=OPTIONS.batch_width,
                                                 session=sess, x_pl=x_pl, skip=OPTIONS.skip, part=channelRange)
                    folder = OPTIONS.checkpoint_dir + modelName + "/"
                    np.save(folder + "sigmas.npy", sigmas)
                else:
                    _, values, vals, params = swo.compare_history(model, modelName, dataLength=OPTIONS.batch_width,
                                                                  session=sess, x_pl=x_pl, skip=OPTIONS.skip,
                                                                  plot_results=False, fullTest=OPTIONS.full_test)

    if OPTIONS.compare and not OPTIONS.calculate_gradient:
        with tf.Session(config=getTfConfig()) as sess:
            model, predictOp, x_pl, _ = setupNetwork(sess)
            swo = inst.get_swaptiongen(getIrModel(), OPTIONS.currency, OPTIONS.irType)
            _, values, vals, params = swo.compare_history(model, modelName, dataLength=OPTIONS.batch_width,
                                                          session=sess, x_pl=x_pl, skip=OPTIONS.skip,
                                                          plot_results=False, fullTest=OPTIONS.full_test)

            np.save(OPTIONS.checkpoint_dir + modelName + "Values.npy", values)
            np.save(OPTIONS.checkpoint_dir + modelName + "Vals.npy", vals)
            np.save(OPTIONS.checkpoint_dir + modelName + "Params.npy", params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action='store_true', help='Train a model')
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
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
    parser.add_argument('-fc', '--fullyConnectedNodes', nargs='+', default=FULLY_CONNECTED_NODES_DEFAULT,
                        help='Comma separated list of number of dense layer depth')
    parser.add_argument('-ar', '--architecture', nargs='+', default=DEFAULT_ARCHITECTURE,
                        help='Comma separated list of characters c->convLayer, l->lstm, f->flatten, d->dense')
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
    parser.add_argument('-vold', '--volFileName', type=str, default=VOL_DATA_DIR_DEFAULT, help='Volatility directory')
    parser.add_argument('-ir', '--irType', type=str, default=IR_DEFAULT, help='Interest rate type')
    parser.add_argument('-ccy', '--currency', type=str, default=CURRENCY_DEFAULT, help='Swaption currency')
    parser.add_argument('-nn', '--nn_model', type=str, default='cnn',
                        help='Type of model. Possible options: cnn and lstm')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    parser.add_argument('-m', '--model', type=str, default='hullwhite', help='Interest rate model')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate history')
    parser.add_argument('--calibrate_sigma', action='store_true', help='Calibrate only sigma')
    parser.add_argument('-hs', '--historyStart', type=str, default=0, help='History start')
    parser.add_argument('-he', '--historyEnd', type=str, default=-1, help='History end')
    parser.add_argument('--compare', action='store_true', help='Run comparison with nn ')
    parser.add_argument('--print_frequency', type=int, default=50, help='Frequency of epoch printing')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip n first dates in history comparison')
    parser.add_argument('-pp', '--pipeline', type=str, default="", help='Pipeline path')
    parser.add_argument('--scaler', type=str, default='minmax', help='Scaler')
    parser.add_argument('-ds', '--decay_steps', type=int, default=3000, help='Decay steps')
    parser.add_argument('-dr', '--decay_rate', type=float, default=0.0, help='Decay rate')
    parser.add_argument('--decay_staircase', action='store_true', help='Decay rate')
    parser.add_argument('--gpu_memory_fraction', type=float, default=GPU_MEMORY_FRACTION,
                        help='Percentage of gpu memory to use')
    parser.add_argument('-cl', '--use_calibration_loss', action='store_true', help='Use nn calibration loss')
    parser.add_argument('-ps', '--predictiveShape', nargs='+', default=None,
                        help='Comma separated list of numbers for the input and output depth of the nn')
    parser.add_argument('-cr', '--channel_range', nargs='+', default=['44'],
                        help='Comma separated list specifying the channel range from the data')
    parser.add_argument('--with_gradient', action='store_true',
                        help='Computes partial derivative wrt the neural network input')
    parser.add_argument('-up', '--use_pipeline', action='store_true',
                        help='Use of pipeline for pre-processing')
    parser.add_argument('--full_test', action='store_true', help='Calibrate history with new starting points++')
    parser.add_argument('--suffix', type=str, default="", help='Custom string identifier for modelName')
    parser.add_argument('--target', type=str, default=None, help='Use specific data target')
    parser.add_argument('--exportForwardRates', action='store_true',
                        help='Calculate and save spot rates to forward rates')
    parser.add_argument('-cg', '--calculate_gradient', action='store_true',
                        help='Imports saved nn weights and calculates the gradient wrt the input')
    parser.add_argument('-fd', '--futureIncrement', type=int, default=365,
                        help='Future reference count of days after initial reference day')
    parser.add_argument('--use_cpu', action='store_true', help='Use cpu instead of gpu')
    parser.add_argument('--no_transform', action='store_true', help="Don't transform data with pipeline")
    parser.add_argument('--extend_training', action='store_true', help="Extend training steps")

    OPTIONS, unparsed = parser.parse_known_args()
    predShape = ''
    if (OPTIONS.predictiveShape is not None):
        cR = '_'.join(OPTIONS.channel_range) if type(OPTIONS.channel_range) == list else str(OPTIONS.channel_range)
        predShape = str('Ps' + '_'.join(OPTIONS.predictiveShape) + '_' + cR + '_')

    modelName = OPTIONS.suffix + predShape + OPTIONS.nn_model + "_A" + ''.join(OPTIONS.architecture) + "_w" + str(
        OPTIONS.batch_width) + "_v" + str(OPTIONS.conv_vol_depth) + "_ir" + str(OPTIONS.conv_ir_depth)
    tf.app.run()
