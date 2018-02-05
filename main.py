import argparse, os, time, sys, pdb, simulate
import tensorflow as tf
from graphHandler import GraphHandler
from models.neuralNet import NeuralNet
import models.SwaptionGenerator as swg
import models.IRCurve as irc
from utils.ahUtils import *
from utils.dataHandler import *
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import utils.customUtils as cu
from utils.FunctionTransformer import FunctionTransformerWithInverse

# region NNDefaultConstants
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
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'adagrad'
NUMBER_OF_NODES = ['2']
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

IR_MODEL = {'hullwhite': hullwhite_analytic,
            'g2': g2,
            'g2_local': g2_local}
# endregion optionDictionaries

OPTIONS = None
optionDict = None
modelName = None
optionList = None
tf.set_random_seed(42)
np.random.seed(42)


def buildNN(dataHandler, swaptionGen=None, chainedModel=None):
    if (optionDict['architecture'][0] == 'd' or 'l' in optionDict['architecture'][0] or 'g' in
            optionDict['architecture'][0]):
        dataHandler.forceSimplify()
    testX, testY = dataHandler.getTestData()
    chained_pl = None
    if chainedModel is not None:
        chained_pl = chainedModel['placeholder']

    print("Input dims" + str(testX.shape))
    if (int(optionDict['nodes'][len(optionDict['nodes']) - 1]) != testY.shape[1]):
        print("LAST fcNodes DIFFERENT SHAPE FROM DATA TARGET")
        print("Aligning...")
        optionDict['nodes'][len(optionDict['nodes']) - 1] = testY.shape[1]
    print("Output dims" + str((None, testY.shape[1])))
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
    outPP, inPP = getPipelines(optionDict)
    try:
        activation = [ACTIVATION_DICT[act] for act in optionDict['activation']]
    except:
        activation = [ACTIVATION_DICT[ACTIVATION_DEFAULT]]
        pass

    nn = NeuralNet(volChannels=optionDict['conv_vol_depth'], irChannels=optionDict['conv_ir_depth'],
                   poolingLayerFlag=poolingFlag, inPipeline=inPP, pipeline=outPP,
                   architecture=optionDict['architecture'], units=optionDict['nodes'],
                   activationFunctions=activation, chainedModel=chainedModel,
                   calibrationFunc=swaptionGen.calibrate if swaptionGen is not None else None)

    pred = nn.inference(x_pl, chained_pl)
    tf.add_to_collection("predict", pred)
    loss = nn.loss(pred, y_pl)
    tf.add_to_collection("loss", loss)

    return dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline, nn


def getOptimizerOperation(loss, learningRate, global_step):
    optimizer = OPTIMIZER_DICT[optionDict['optimizer']](learning_rate=learningRate)
    opt = optimizer.minimize(loss, global_step=global_step)
    return opt


def getLearningrate(decay_rate, global_step):
    if (decay_rate > 0):
        learningRate = tf.train.exponential_decay(learning_rate=optionDict['learning_rate'],
                                                  global_step=global_step,
                                                  decay_steps=optionDict['decay_steps'],
                                                  decay_rate=optionDict['decay_rate'],
                                                  staircase=optionDict['decay_staircase'])
    else:
        learningRate = optionDict['learning_rate']

    return learningRate


def trainNN(dataHandler, network, loss, pred, x_pl, y_pl, testX, testY, chainedModel=None):
    # With chained models use already transformed/preprocessed data to avoid some complexity
    mergedSummaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    inputPipeline = outPipeline = None
    try:
        global_step = tf.Variable(0, trainable=False)
        prevTestLoss = 1
        checkpointFolder = optionDict['checkpoint_dir'] + modelName + "/"
        timestamp = modelName + ''.join(str(dt.datetime.now().timestamp()).split('.'))

        learningRate = getLearningrate(optionDict['decay_rate'], global_step)
        opt = getOptimizerOperation(loss, learningRate, global_step)
        inputPipeline, outPipeline = dataHandler.initializePipelines(inputPipeline=network.inputPipeline,
                                                                     outPipeline=network.pipeline)

        gradient = None
        if (optionDict['with_gradient']):
            gradient = tf.gradients(pred, x_pl)

        with tf.Session(config=getTfConfig()) as sess:
            chained_pl = chainedDH = None  # prevent IDE from being annoying
            if (chainedModel is not None):
                chainedDH, chainedModel = setupChainedModel(chainedModel, useDataHandler=True)
                gh = chainedModel['model']
                chained_pl = chainedModel['placeholder']
                # assert dataHandler.dataFileName == chainedDH.dataFileName, "The chained models MUST use the same data source"
                assert dataHandler.sliding == chainedDH.sliding, "The chained models MUST use same sliders"
                assert dataHandler.randomSpliting == False and chainedDH.randomSpliting == False, "No random splitting"
                assert dataHandler.testDataPercentage == chainedDH.testDataPercentage, "Same test percentage"

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(optionDict['log_dir'] + '/train' + timestamp, sess.graph)
            test_writer = tf.summary.FileWriter(optionDict['log_dir'] + '/test' + timestamp, sess.graph)

            ttS = time.time()
            max_steps = optionDict['max_steps'] + 1
            epoch = 0
            # pdb.set_trace()

            while epoch < max_steps:
                ttS = time.time() if epoch % optionDict['print_frequency'] == 1 else ttS
                batch_x, batch_y = dataHandler.getNextBatch(randomDraw=False)
                if (chainedModel is not None):
                    chained_train_x, _ = chainedDH.getNextBatch()
                    cOut = gh.run(op=gh.gradientOp, data=chained_train_x)
                    chainedInput = gh.model.derivationProc(cOut, gh.model.irChannels + gh.model.volChannels,
                                                           chained_train_x.shape)
                    if (inputPipeline is not None):
                        chainedInput = inputPipeline.transform(chainedInput)
                    _, out, merged_sum = sess.run([opt, loss, mergedSummaries],
                                                  feed_dict={x_pl: batch_x, y_pl: batch_y, chained_pl: chainedInput})
                else:
                    _, out, merged_sum = sess.run([opt, loss, mergedSummaries],
                                                  feed_dict={x_pl: batch_x, y_pl: batch_y})
                if epoch % optionDict['print_frequency'] == 0:
                    elapsedTime = time.time() - ttS
                    train_writer.add_summary(merged_sum, epoch)
                    train_writer.flush()
                    lrPrint = learningRate if type(learningRate) is float else learningRate.eval()
                    print("====================================================================================")
                    print("Epoch:", '%06d' % (epoch), "Learning rate", '%06f' % (lrPrint), "loss=",
                          "{:.8f}".format(out), "Elapsed time: %03f" % elapsedTime)
                if epoch % optionDict['test_frequency'] == 0 and epoch > 0:
                    if (epoch == OPTIONS.test_frequency):
                        if (outPipeline is not None):
                            print("Transforming testY")
                            testY = outPipeline.transform(testY)
                        if (inputPipeline is not None):
                            print("Transforming testX")
                            if (len(testX.shape) == 3):
                                for i in range(np.asarray(testX).shape[2]):
                                    testX[:, :, i] = inputPipeline.transform(testX[:, :, i])

                            elif (len(testX.shape) == 4):
                                for i in range(np.asarray(testX).shape[3]):
                                    testX[:, 0, :, i] = inputPipeline.transform(testX[:, 0, :, i])
                    # pdb.set_trace()
                    if (chainedModel is not None):
                        chained_test_x, _ = chainedDH.getTestData()
                        cOut = gh.run(op=gh.gradientOp, data=chained_test_x)
                        chainedInput = gh.model.derivationProc(cOut, gh.model.irChannels + gh.model.volChannels,
                                                               chained_test_x.shape)

                        if (inputPipeline is not None):
                            chainedInput = inputPipeline.transform(chainedInput)
                        out, merged_sum = sess.run([loss, mergedSummaries],
                                                   feed_dict={x_pl: testX, y_pl: testY, chained_pl: chainedInput})
                    else:
                        out, merged_sum = sess.run([loss, mergedSummaries], feed_dict={x_pl: testX, y_pl: testY})

                    if (out + 0.00001 <= prevTestLoss and optionDict['extend_training']):
                        prevTestLoss = out
                        if (max_steps - epoch <= optionDict['test_frequency']):
                            max_steps += optionDict['test_frequency'] + 1
                    test_writer.add_summary(merged_sum, epoch)
                    test_writer.flush()
                    print("Test set:" "loss=", "{:.10f}".format(out))
                if epoch % optionDict['checkpoint_freq'] == 0 and epoch > 0:
                    saver.save(sess, checkpointFolder + str(epoch))
                epoch += 1

            if (gradient is not None):
                derivative = sess.run(gradient, feed_dict={x_pl: testX})
                der = sess.run(gradient, feed_dict={x_pl: np.vstack((dataHandler.trainData['input'], testX))})
                folder = optionDict['checkpoint_dir'] + modelName
                np.save(folder + "/" + "fullRawDerivatives.npy", der)
        #                transformDerivatives(derivative, dataHandler, testX, folder=checkpointFolder)

        optionDict['input_pipeline'] = savePipeline(inputPipeline, 'in')
        optionDict['pipeline'] = savePipeline(outPipeline, 'out')
        opts = get_options(False, modelDir=checkpointFolder + str(optionDict['max_steps']))

        if len(testX.shape) == 2:
            xShape = (None, testX.shape[1])
        elif (len(testX.shape) == 3):
            xShape = (None, testX.shape[1], testX.shape[2])
        elif (len(testX.shape) == 4):
            xShape = (None, 1, testX.shape[2], testX.shape[3])
        opts["input_dims"] = xShape

        if (gradient is not None):
            opts["output_dims"] = (None, 1)
        else:
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


def setupChainedModel(chainedModelDict, useDataHandler=False):
    dh = None
    if (useDataHandler):
        dh = setupDataHandler(chainedModelDict['options'], allowPredictiveTransformation=False)
    gradFlag = chainedModelDict['options']['with_gradient']
    gh = setupNetwork(chainedModelDict['options'], gradientFlag=gradFlag)
    if (chainedModelDict['options']['input_pipeline'] != ""):
        pipelineList = cu.loadSavedScaler(chainedModelDict['options']['input_pipeline'])
        gh.model.setInputPipelineList(pipelineList)
    chainedModelDict['model'] = gh
    return dh, chainedModelDict


def transformDerivatives(derivative, dataHandler, testX, folder=None, save=True):
    der = cu.transformDerivatives(derivative, dataHandler.channelStart, dataHandler.channelEnd, testX.shape)
    if (save):
        if (folder is None):
            folder = optionDict['checkpoint_dir'] + modelName
        np.save(folder + "/" + "testDerivatives.npy", der)

    return der


def setupDataHandler(options, allowPredictiveTransformation=True, testPercentage=DEFAULT_TEST_DATA_PERCENTAGE):
    # to avoid cutting the data for training set allowPredictiveTransformation False (e.g. chained model)
    if (type(options) == argparse.Namespace):
        options = vars(options)
    if (options['predictiveShape'] is not None):
        if (options['volFileName'] is not None):
            mode = 'vol'
        elif (options['irFileName'] is not None):
            mode = 'ir'
        else:
            raise ValueError('File name is not correct')

        predictiveShape = (mode, options['predictiveShape'], options['channel_range'], allowPredictiveTransformation)
        specialFilePrefix = "_M" + mode + ''.join(options['predictiveShape'])
    else:
        predictiveShape = None
        specialFilePrefix = ''
    if ('testDataPercentage' not in options):
        options['testDataPercentage'] = testPercentage
    dataFileName = options['volFileName'] if options['volFileName'] is not None else options['irFileName']
    dataHandler = DataHandler(dataFileName=dataFileName, batchSize=options['batch_size'], width=options['batch_width'],
                              testDataPercentage=options['testDataPercentage'],
                              volDepth=int(options['data_vol_depth']), irDepth=int(options['data_ir_depth']),
                              useDataPointers=False, save=options['saveProcessedData'],
                              specialFilePrefix=specialFilePrefix, predictiveShape=predictiveShape,
                              targetDataMode=options['target'])
    if (options['processedData']):
        fileList = dataHandler.findTwinFiles(dataFileName)
        simplify = False
        if (options['architecture'][0] == 'd' or 'l' in options['architecture'][0] or 'g' in options['architecture'][
            0]):
            simplify = True
        dataHandler.delegateDataDictsFromFile(fileList, simplify)

    return dataHandler


def get_options(printFlag=True, modelDir=None):
    options = optionDict
    for key, value in optionDict.items():
        if (key == 'model_dir' and modelDir is not None):
            value = modelDir
            options[key] = modelDir
        item = key + ' : ' + str(value)
        # array.append((key, value))
        if (printFlag):
            print(item)
    return options


def initDirectories():
    # Make directories if they do not exists yet
    if not tf.gfile.Exists(optionDict['log_dir']):
        tf.gfile.MakeDirs(optionDict['log_dir'])
    if not tf.gfile.Exists(optionDict['data_dir']):
        tf.gfile.MakeDirs(optionDict['data_dir'])
    if not tf.gfile.Exists(optionDict['checkpoint_dir']):
        tf.gfile.MakeDirs(optionDict['checkpoint_dir'])


def getIrModel():
    if (str(optionDict['model']) in IR_MODEL):
        return IR_MODEL[optionDict['model']]
    else:
        raise RuntimeError("Unknown IR model")


def getPipelines(options):
    inPP = outPP = None
    if (options['use_pipeline']):
        try:
            if (options['pipeline'] is not ""):
                pipelinePath = options['pipeline']
            else:
                pipelinePath = options['checkpoint_dir'] + modelName + "/pipeline.pkl"
            if (not os.path.exists(pipelinePath)):
                pipelinePath = None
            outPP = buildPipeLine(options['transform'], options['scaler'], pipelinePath)
        except:
            pass
    if (options['use_input_pipeline']):
        try:
            if (options['input_pipeline'] is not ""):
                pipelinePath = options['input_pipeline']
            else:
                pipelinePath = options['checkpoint_dir'] + modelName + "/input_pipeline.pkl"
                if (not os.path.exists(pipelinePath)):
                    pipelinePath = None
            inPP = buildPipeLine(options['in_transform'], options['in_scaler'], pipelinePath)
        except:
            pass
    return outPP, inPP


def buildPipeLine(transform, scaler, fName=None):
    if (fName is None):
        irModel = getIrModel()
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
            pipeline = cu.loadSavedScaler(optionDict['input_pipeline'])
    return pipeline


def savePipeline(pipeline, mode):
    pipelinePath = ""
    if (pipeline is not None):
        pipelinePath = optionDict['checkpoint_dir'] + modelName
        if (not os.path.exists(pipelinePath)):
            os.makedirs(pipelinePath)
        if (mode == 'in'):
            pref = 'input_'
        else:
            pref = 'out_'
        pipelinePath = pipelinePath + '/' + pref + "pipeline.pkl"
        joblib.dump(pipeline, pipelinePath)

    return pipelinePath


def getTfConfig():
    if (optionDict['use_cpu']):
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpuMem = tf.GPUOptions(per_process_gpu_memory_fraction=optionDict['gpu_memory_fraction'])
        config = tf.ConfigProto(gpu_options=gpuMem)

    return config


def setupNetwork(options, chainedDict=None, gradientFlag=False, prefix="", inMultipleNetsIndex=None):
    """
    :param options:
    :param chainedDict
    :param gradientFlag:
    :param prefix:
    :return: GraphHandler
    """
    if (type(options) == argparse.Namespace):
        options = vars(options)
    fName, directory = cu.splitFileName(options['model_dir'])
    gh = GraphHandler(directory, options['nn_model'], sessConfig=getTfConfig(), chainedPrefix=prefix)
    outPipeline, inPipeline = getPipelines(options)
    gh.importSavedNN(fName, gradientFlag=gradientFlag)
    gh.buildModel(options, chained=chainedDict, outPipeline=outPipeline, inPipeline=inPipeline,
                  inMultipleNetsIndex=None)
    return gh


def buildModelName(ps, cr, cm, suff, nn, arch, bw, cvd, cid):
    predShape = ''
    if (ps is not None):
        cR = '_'.join(cr) if type(cr) == list else str(cr)
        predShape = str('Ps' + '_'.join(ps) + '_' + cR + '_')
    if (cm is not None):
        preSuffix = 'chained_'
    else:
        preSuffix = ""
    mN = preSuffix + suff + predShape + nn + "_A" + ''.join(arch) + "_w" + str(bw) + "_v" + str(cvd) + "_ir" + str(cid)

    return mN


def loadChained(options):
    if (type(options) == argparse.Namespace):
        options = vars(options)
    if (options['chained_model'] is not None):
        fName, directory = cu.splitFileName(options['chained_model'])
        chainedOptions = cu.load_obj(directory + '/options.pkl')
        if (options['chained_pipeline'] != ""):
            chainedOptions['input_pipeline'] = options['chained_pipeline']
        # DO NOT CONFUSE, chained['placeholder'] keeps the reference to the placeholder of the
        # forward chained network i.e. the output of chained['model'] network
        # and is delegated in this dictionary when building the new network (in new network's graph).
        # 'model' key is expected to carry an GraphHandler item of the network
        chained_pl = tf.placeholder(tf.float32, shape=chainedOptions['output_dims'], name="chainedx_pl")
        # When loading models chained_pl MUST be the reference to the placeholder loaded in the "chained graph"
        # If the model is created in this scope then placeholder is already in the "chained graph" = default graph
        # this is handled in GraphHandler.buildModel()
        chained = {"output_dims": chainedOptions['output_dims'], 'options': chainedOptions, 'model': None,
                   'placeholder': chained_pl}
    else:
        chained = None

    return chained


def loadMultipleNetworks():
    ghList = []
    channelRange = []
    pipelineList = None
    for j in range(len(optionList)):
        optionList[j]['conv_ir_depth'] = 1
        temp = setupNetwork(options=optionList[j], gradientFlag=True, inMultipleNetsIndex=j)
        if (optionList[j]['use_input_pipeline'] and optionList[j]['input_pipeline'] is not None):
            pipelineList = cu.loadSavedScaler(optionList[1]['input_pipeline'])
            temp.model.setInputPipelineList(pipelineList)
        if (len(optionList[j]['channel_range']) > 1):
            channelRange.append([int(optionList[j]['channel_range'][0]),
                                 int(optionList[j]['channel_range'][1])])
        else:
            channelRange.append([0, int(optionList[j]['channel_range'][0])])
        ghList.append(temp)
    return ghList


def main(_):
    _ = get_options()
    initDirectories()
    swo = None
    if (optionDict['weight_reg_strength'] is None):
        optionDict['weight_reg_strength'] = 0.0
    swg.setDataFileName(optionDict['data_dir'])

    if optionDict['calibrate']:
        swo = swg.get_swaptiongen(getIrModel(), currency=optionDict['currency'], irType=optionDict['irType'],
                                  volFileName=optionDict['volFileName'], irFileName=optionDict['irFileName'])
        swo.calibrate_history(start=int(optionDict['historyStart']), end=int(optionDict['historyEnd']))

    if optionDict['exportForwardRates']:
        exportPath = './exports/' + optionDict['suffix'] + 'fwCurves' + "_fDays" + str(
            optionDict['futureIncrement']) + ".csv"
        if (not os.path.isfile(exportPath)):
            ir = irc.getIRCurves(getIrModel(), currency=optionDict['currency'], irType=optionDict['irType'],
                                 irFileName=optionDict['irFileName'])
            # dateInDays = optionDict['irType'] if optionDict['dayDict']
            # ir.calcForward(path=exportPath, futureIncrementInDays=optionDict['futureIncrement'])
            ir.calcThetaHW(path=exportPath)
        else:
            print("File already exists")

    if optionDict['is_train']:
        dh = setupDataHandler(optionDict)
        if optionDict['nn_model'] == 'cnn' or optionDict['nn_model'] == 'lstm':
            chained = loadChained(optionDict)
            if optionDict['retrain']:
                gh = setupNetwork(options=optionDict, gradientFlag=True)
                with gh.graph.as_default():
                    global_step = gh.graph.get_tensor_by_name('Variable:0')
                    learningRate = getLearningrate(optionDict['decay_rate'], global_step)
                    opt = getOptimizerOperation(gh.loss, learningRate, global_step)
                gh.simpleRetrain(dh, learningRate, opt, optionDict, modelName)
            else:
                dataHandler, loss, pred, x_pl, y_pl, testX, testY, pipeline, nn = buildNN(dh, swaptionGen=swo,
                                                                                          chainedModel=chained)
                outPipeline, inPipeline = getPipelines(optionDict)
                nn.inputPipeline = inPipeline
                nn.pipeline = outPipeline
                trainNN(dataHandler, nn, loss, pred, x_pl, y_pl, testX, testY,
                        chainedModel=nn.chainedModel)
        else:
            raise ValueError("--train_model argument can be lstm or cnn")

    if optionDict['simulate']:
        swo = swg.get_swaptiongen(getIrModel(), optionDict['currency'], optionDict['irType'])
        if (optionList is not None):
            gh = loadMultipleNetworks()
        else:
            gh = setupNetwork(options=optionDict, gradientFlag=True)
            if (optionDict['use_input_pipeline'] and optionDict['input_pipeline'] is not None):
                pipelineList = cu.loadSavedScaler(optionDict['input_pipeline'])
                gh.model.setInputPipelineList(pipelineList)
        simulate.runSimulations(gh, modelName, swo, optionDict['batch_width'])

    if optionDict['calculate_gradient']:
        if optionDict['calibrate_sigma']:
            swo = swg.get_swaptiongen(getIrModel(), optionDict['currency'], optionDict['irType'])
            channelRange = -1
            if (optionList is not None):
                gh = loadMultipleNetworks()
            else:
                gh = setupNetwork(options=optionDict, gradientFlag=True)
                if (optionDict['use_input_pipeline'] and optionDict['input_pipeline'] is not None):
                    pipelineList = cu.loadSavedScaler(optionDict['input_pipeline'])
                    gh.model.setInputPipelineList(pipelineList)

                if (len(optionDict['channel_range']) > 1):
                    channelRange = [int(optionDict['channel_range'][0]), int(optionDict['channel_range'][1])]
                else:
                    channelRange = [0, int(optionDict['channel_range'][0])]

            sigmas = swo.calibrate_sigma(gh, modelName, dataLength=optionDict['batch_width'],
                                         skip=optionDict['skip'], part=channelRange)
            ffolder = optionDict['checkpoint_dir'] + modelName + "/"
            try:
                np.save(ffolder + "sigmas.npy", sigmas)
            except:
                np.save(optionDict['suffix'] + 'sigmas.npy', sigmas)
        else:
            gh = setupNetwork(options=optionDict, gradientFlag=True)
            dh = setupDataHandler(optionDict, allowPredictiveTransformation=True, testPercentage=0)
            testX, _ = dh.getTestData()
            if (len(testX.shape) > 2):
                testX = testX[:, :, :, :optionDict['conv_ir_depth']]
            dh.getNextBatch()
            if (len(testX.shape) > 2):
                trainX = dh.trainData['input'][:, :, :, :optionDict['conv_ir_depth']]
            else:
                trainX = dh.trainData['input']
            inPut = np.vstack((trainX, testX))
            # pdb.set_trace()
            deriv = gh.run(inPut, gh.gradientOp)
            path = optionDict['checkpoint_dir'] if optionDict['checkpoint_dir'] != CHECKPOINT_DIR_DEFAULT else None
            transformDerivatives(deriv, dh, inPut, path)

    if optionDict['compare']:
        with tf.Session(config=getTfConfig()) as sess:
            chained = loadChained(optionDict)
            if (chained is not None):
                _, chained = setupChainedModel(chained)
            gh = setupNetwork(optionDict, chainedDict=chained, prefix="chained" if (chained is not None) else "")
            if (optionDict['use_input_pipeline'] and optionDict['input_pipeline'] != ""):
                pipelineList = cu.loadSavedScaler(optionDict['input_pipeline'])
                gh.model.setInputPipelineList(pipelineList)

            swo = swg.get_swaptiongen(getIrModel(), optionDict['currency'], optionDict['irType'])
            _, values, vals, params = swo.compare_history(gh, modelName, dataLength=optionDict['batch_width'],
                                                          skip=optionDict['skip'], plot_results=False,
                                                          fullTest=optionDict['full_test'])
            try:
                np.save(optionDict['checkpoint_dir'] + modelName + "/Values.npy", values)
                np.save(optionDict['checkpoint_dir'] + modelName + "/Vals.npy", vals)
                np.save(optionDict['checkpoint_dir'] + modelName + "/Params.npy", params)
            except:
                np.save("Values.npy", values)
                np.save("Vals.npy", vals)
                np.save("Params.npy", params)


def loadOptions(path):
    file, model = cu.splitFileName(path)
    opts = cu.load_obj(model + '/options.pkl')
    return opts


if __name__ == '__main__':
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
    parser.add_argument('-ipp', '--input_pipeline', type=str, default="", help="Pipeline path for input data")
    parser.add_argument('-cpp', '--chained_pipeline', type=str, default="", help='Pipeline path of chained model')
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
    parser.add_argument('--suffix', type=str, default="", help='Custom string identifier for modelName')
    parser.add_argument('--target', type=str, default=None, help='Use specific data target')
    parser.add_argument('--exportForwardRates', action='store_true',
                        help='Calculate and save spot rates to forward rates')
    parser.add_argument('-cg', '--calculate_gradient', action='store_true',
                        help='Imports saved nn weights and calculates the gradient wrt the input')
    parser.add_argument('-fd', '--futureIncrement', type=int, default=365,
                        help='Future reference count of days after initial reference day')
    parser.add_argument('--use_cpu', action='store_true', help='Use cpu instead of gpu')
    parser.add_argument('--transform', action='store_true', help="Transform data with pipeline")
    parser.add_argument('--in_transform', action='store_true', help="Transform input data with pipeline")
    parser.add_argument('--extend_training', action='store_true', help="Extend training steps")
    parser.add_argument('--load_options', action='store_true', help="Load options from file")
    parser.add_argument('-chain', '--chained_model', type=str, default=None,
                        help="Loads model defined in file and chains it with the current nn model")
    parser.add_argument('--load_multiple', type=int, default=0, help="Load multiple models")
    parser.add_argument('--simulate', action='store_true', help="Run simulations")

    OPTIONS, unparsed = parser.parse_known_args()
    if (OPTIONS.load_multiple > 0):
        fileName, model_dir = cu.splitFileName(OPTIONS.model_dir)
        m = str(model_dir).split("_" + str(OPTIONS.nn_model) + "_")
        m[0] = m[0][:len(m) - 5]
        optionList = []
        for i in range(OPTIONS.load_multiple - 1):
            folder = ''.join([m[0], str(i) + "_" + str(i + 1), '_cnn_', m[1]]) + str(fileName)
            optionList.append(loadOptions(folder))
            optionList[i]['model_dir'] = folder

    if (OPTIONS.load_options and OPTIONS.load_multiple == 0):
        try:
            fileName, model_dir = cu.splitFileName(OPTIONS.model_dir)
            optionDict = cu.load_obj(model_dir + '/options.pkl')
            if (OPTIONS.chained_pipeline != ""):
                optionDict['input_pipeline'] = OPTIONS.chained_pipeline
                optionDict['use_input_pipeline'] = True
            optionDict['model_dir'] = OPTIONS.model_dir  # to use specific checkpoint
            optionDict['irFileName'] = OPTIONS.irFileName if OPTIONS.irFileName is not None else optionDict[
                'irFileName']
            # Keep basic operations
            optionDict['calibrate'] = OPTIONS.calibrate
            optionDict['calibrate_sigma'] = OPTIONS.calibrate_sigma
            optionDict['exportForwardRates'] = OPTIONS.exportForwardRates
            optionDict['is_train'] = OPTIONS.is_train
            optionDict['retrain'] = OPTIONS.retrain
            optionDict['compare'] = OPTIONS.compare
            optionDict['calculate_gradient'] = OPTIONS.calculate_gradient
            # Keep calibration related input
            optionDict['currency'] = OPTIONS.currency
            optionDict['irType'] = OPTIONS.irType
            optionDict['simulate'] = OPTIONS.simulate
        except Exception as ex:
            raise Exception("Exception loading option from file:" + str(ex))
    else:
        optionDict = vars(OPTIONS)
    modelName = buildModelName(ps=optionDict['predictiveShape'], cr=optionDict['channel_range'],
                               cm=optionDict['chained_model'], suff=optionDict['suffix'], nn=optionDict['nn_model'],
                               arch=optionDict['architecture'], bw=optionDict['batch_width'],
                               cvd=optionDict['conv_vol_depth'], cid=optionDict['conv_ir_depth'])

    tf.app.run()
