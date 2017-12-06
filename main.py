import argparse
import os
import tensorflow as tf
import numpy as np
from models.cnet import ConvNet
import models.instruments as inst
from dataUtils.dataHandler import DataHandler
import dataUtils.data_utils as du
import pdb
import datetime as dt
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

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
BATCH_WIDTH_DEFAULT = 50
CONVOLUTION_VOL_DEPTH_DEFAULT = '3'
CONVOLUTION_IR_DEPTH_DEFAULT = '2'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
FULLY_CONNECTED_NODES_DEFAULT = ['2']
DEFAULT_ARCHITECTURE = ['c', 'f', 'd']
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
WEIGHT_INITIALIZATION_DICT = {'xavier': tf.contrib.layers.xavier_initializer,  # Xavier initialisation
                              'normal': tf.random_normal_initializer,  # Initialization from a standard normal
                              'uniform': tf.random_normal_initializer}  # Initialization from a uniform distribution

WEIGHT_REGULARIZER_DICT = {'none': tf.contrib.layers.l1_regularizer,  # No regularization
                           'l1': tf.contrib.layers.l1_regularizer,  # L1 regularization
                           'l2': tf.contrib.layers.l2_regularizer}  # L2 regularization

ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.tanh,  # Tanh
                   'sigmoid': tf.sigmoid}  # Sigmoid

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

FLAGS = None
modelName = None
tf.set_random_seed(42)
np.random.seed(42)


def trainLSTM(dataHandler):
    tf.set_random_seed(42)
    np.random.seed(42)
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []


def buildCnn(dataHandler):
    testX, testY = dataHandler.getTestData()
    x_pl = tf.placeholder(tf.float32, shape=(None, 1, testX.shape[2], testX.shape[3]), name="x_pl")
    y_pl = tf.placeholder(tf.float32, shape=(None, testY.shape[1]), name="y_pl")
    # print("Weight initialization: ", FLAGS.weight_init)

    poolingFlag = tf.placeholder(tf.bool)
    cnn = ConvNet(poolingLayerFlag=poolingFlag, architecture=FLAGS.architecture, fcUnits=FLAGS.fullyConnectedNodes)
    pred = cnn.inference(x_pl)
    tf.add_to_collection("predict", pred)
    loss = cnn.loss(pred, y_pl)
    # accuracy=cnn.accuracy(pred,y_pl)

    return dataHandler, loss, x_pl, y_pl, testX, testY


def trainNN(dataHandler, loss, x_pl, y_pl, testX, testY):
    mergedSummaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    try:
        gpuMem = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpuMem)) as sess:
            sess.run(tf.global_variables_initializer())
            timestamp = ''.join(str(dt.datetime.now().timestamp()).split('.'))
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train' + timestamp, sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test' + timestamp, sess.graph)
            pipeline = getPipeLine()
            global_step = tf.Variable(0, trainable=False)
            if (FLAGS.decay_rate > 0):
                learningRate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=global_step,
                                                          decay_steps=FLAGS.decay_steps,
                                                          decay_rate=FLAGS.decay_rate, staircase=FLAGS.decay_staircase)
            else:
                learningRate = FLAGS.learning_rate

            pdb.set_trace()
            optimizer = OPTIMIZER_DICT[FLAGS.optimizer](learning_rate=learningRate)
            opt = optimizer.minimize(loss)
            
            checkpointFolder = FLAGS.checkpoint_dir + modelName + "/"
            for epoch in range(FLAGS.max_steps):
                batch_x, batch_y = dataHandler.getNextBatch(pipeline=pipeline, randomDraw=False)
                global_step = epoch
                _, out, merged_sum = sess.run([opt, loss, mergedSummaries],
                                              feed_dict={x_pl: batch_x, y_pl: batch_y})
                if epoch % FLAGS.print_frequency == 0:
                    train_writer.add_summary(merged_sum, epoch)
                    train_writer.flush()
                    print("====================================================================================")
                    print("Epoch:", '%06d' % (epoch), "Learning rate", '%06f' % (learningRate), "loss=",
                          "{:.6f}".format(out))
                if epoch % FLAGS.test_frequency == 0 and epoch > 0:
                    if (epoch == FLAGS.test_frequency):
                        print("Transforming test")
                        testY = pipeline.transform(testY)

                    out, merged_sum = sess.run([loss, mergedSummaries], feed_dict={x_pl: testX, y_pl: testY})
                    test_writer.add_summary(merged_sum, epoch)
                    test_writer.flush()
                    print("Test set:" "loss=", "{:.6f}".format(out))
                if epoch % FLAGS.checkpoint_freq == 0 and epoch > 0:
                    saver.save(sess, checkpointFolder + str(epoch))
        tf.reset_default_graph()
    except:
        print("Exception during training")
        tf.reset_default_graph()

    return pipeline


def importSavedNN(session, modelPath, fileName):
    saver = tf.train.import_meta_graph(modelPath + "/" + fileName + ".meta", clear_devices=True)
    graph = tf.get_default_graph()
    check = tf.train.get_checkpoint_state(modelPath)
    x_pl = graph.get_tensor_by_name("x_pl:0")
    # x_pl = tf.placeholder(tf.float32, shape=(None, 1, 50, 5))
    session.run(tf.global_variables_initializer())
    saver.restore(session, check.model_checkpoint_path)
    return tf.get_collection("predict")[0], x_pl


def setupDataHandler():
    dataHandler = DataHandler(dataFileName=FLAGS.volFileName, batchSize=FLAGS.batch_size, width=FLAGS.batch_width,
                              volDepth=int(FLAGS.conv_vol_depth[0]), irDepth=int(FLAGS.conv_ir_depth[0]),
                              save=FLAGS.saveProcessedData)
    if (FLAGS.processedData):
        fileList = dataHandler.findTwinFiles(FLAGS.volFileName)
        dataHandler.delegateDataDictsFromFile(fileList)

    return dataHandler


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def initDirectories():
    # Make directories if they do not exists yet
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def getIrModel():
    if (str(FLAGS.model) in IR_MODEL):
        return IR_MODEL[FLAGS.model]
    else:
        raise RuntimeError("Unknown IR model")


def getPipeLine(fileName=None):
    if (fileName is None):
        irModel = getIrModel()
        transformFunc = inst.FunctionTransformerWithInverse(func=irModel["transformation"],
                                                            inv_func=irModel["inverse_transformation"])
        pipeline = Pipeline([('funcTrm', transformFunc), ('scaler', MinMaxScaler())])
    else:
        pipeline = joblib.load(fileName)

    return pipeline


def main(_):
    print_flags()
    initDirectories()
    swo = None
    if (FLAGS.weight_reg_strength is None):
        FLAGS.weight_reg_strength = 0.0
    inst.setDataFileName(FLAGS.data_dir)
    if FLAGS.calibrate:
        pdb.set_trace()
        swo = inst.get_swaptiongen(getIrModel(), FLAGS.currency, FLAGS.irType, volFileName=FLAGS.volFileName,
                                   irFileName=FLAGS.irFileName)
        swo.calibrate_history(start=int(FLAGS.historyStart), end=int(FLAGS.historyEnd))

    if FLAGS.is_train:
        dh = setupDataHandler()
        if FLAGS.nn_model == 'cnn':
            dataHandler, loss, x_pl, y_pl, testX, testY = buildCnn(dh)
            pipeline = trainNN(dataHandler, loss, x_pl, y_pl, testX, testY)
            if (pipeline is not None):
                pipelinePath = FLAGS.checkpoint_dir + FLAGS.nn_model + "_A" + ''.join(
                    FLAGS.architecture).lower() + "_w" + str(FLAGS.batch_width)
                if (not os.path.exists(pipelinePath)):
                    os.makedirs(pipelinePath)
                joblib.dump(pipeline, pipelinePath + "/pipeline.pkl", compress=1)

        elif FLAGS.nn_model == 'lstm':
            trainLSTM(dh)
        else:
            raise ValueError("--train_model argument can be lstm or cnn")

    if FLAGS.compare:
        # if (FLAGS.nn.lower() == 'cnn'):
        # loadCnn(FLAGS.conv_vol_depth, FLAGS.conv_ir_depth)
        gpuMem = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpuMem)) as sess:
            fileName = ''.join(re.findall(r'(/)(\w+)', FLAGS.model_dir).pop())
            directory = FLAGS.model_dir.split(fileName)[0]
            predictOp, x_pl = importSavedNN(sess, directory, fileName)
            if (FLAGS.nn_model.lower() == 'cnn'):
                pipelinePath = FLAGS.checkpoint_dir + '/' + modelName + "/pipeline.pkl"
                model = ConvNet(predictOp=predictOp, pipeline=getPipeLine(pipelinePath))
            elif (FLAGS.nn_model.lower() == 'lstm'):
                # model = LSTM(predictOp = predictOp)
                pass
            swo = inst.get_swaptiongen(getIrModel(), FLAGS.currency, FLAGS.irType)
            _, values, vals, params = swo.compare_history(model, modelName, dataLength=FLAGS.batch_width, session=sess,
                                                          x_pl=x_pl, skip=FLAGS.skip, plot_results=False)

            np.save(FLAGS.checkpoint_dir + modelName + "Values.npy", values)
            np.save(FLAGS.checkpoint_dir + modelName + "Vals.npy", vals)
            np.save(FLAGS.checkpoint_dir + modelName + "Params.npy", params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action='store_true', help='Train a model')
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('-bw', '--batch_width', type=int, default=BATCH_WIDTH_DEFAULT,
                        help='Batch width')
    parser.add_argument('-cvd', '--conv_vol_depth', type=str, default=CONVOLUTION_VOL_DEPTH_DEFAULT,
                        help='Comma separated list of number of convolution depth for volatilities in each layer')
    parser.add_argument('-cid', '--conv_ir_depth', type=str, default=CONVOLUTION_IR_DEPTH_DEFAULT,
                        help='Comma separated list of number of convolution depth for interest rate in each layer')
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
    parser.add_argument('--activation', type=str, default=ACTIVATION_DEFAULT,
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
    parser.add_argument('-hs', '--historyStart', type=str, default=0, help='History start')
    parser.add_argument('-he', '--historyEnd', type=str, default=-1, help='History end')
    parser.add_argument('--compare', action='store_true', help='Run comparison with nn ')
    parser.add_argument('--print_frequency', type=int, default=50, help='Frequency of epoch printing')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip n first dates in history comparison')
    parser.add_argument('-pp', '--pipeline', type=str, default="", help='Pipeline path')
    parser.add_argument('-ds', '--decay_steps', type=int, default=0, help='Decay steps')
    parser.add_argument('-dr', '--decay_rate', type=float, default=0.0, help='Decay rate')
    parser.add_argument('--decay_staircase', action='store_true', help='Decay rate')

    FLAGS, unparsed = parser.parse_known_args()
    modelName = FLAGS.nn_model + "_A" + ''.join(FLAGS.architecture) + "_w" + str(FLAGS.batch_width)
    tf.app.run()
