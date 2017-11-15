import argparse
import tensorflow as tf
import numpy as np
from models.cnet import ConvNet
import models.instruments as inst
from dataUtils.readData import DataHandler
import dataUtils.data_utils as du
import pdb

# region NNDefaultConstants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 50
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.2
TEST_FREQUENCY_DEFAULT = 0
CHECKPOINT_FREQ_DEFAULT = 200
DNN_HIDDEN_UNITS_DEFAULT = '100'
BATCH_WIDTH_DEFAULT = 50
CONVOLUTION_VOL_DEPTH_DEFAULT = '3'
CONVOLUTION_IR_DEPTH_DEFAULT = '2'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
# endregion NNDefaultConstants

# region QLDefaultConstants
CURRENCY_DEFAULT = 'EUR'
IR_DEFAULT = 'EURIBOR'
# endregion QLDefaultConstants

# region defaultDirectories
DATA_DIR_DEFAULT = 'data/data.h5'
VOL_DATA_DIR_DEFAULT = 'data/_vol.npy'
IR_DATA_DIR_DEFAULT = 'data/_ir.npy'
PARAMS_DATA_DIR_DEFAULT = 'data/_params.npy'
# Directory for tensorflow
LOG_DIR_DEFAULT = 'logs/'
CHECKPOINT_DIR_DEFAULT = 'checkpoints/'
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


def trainLSTM():
    tf.set_random_seed(42)
    np.random.seed(42)
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []


def trainCNN():
    tf.set_random_seed(42)
    np.random.seed(42)

    dataHandler = DataHandler(dataFileName=FLAGS.volFileName, batchSize=FLAGS.batch_size, width=FLAGS.batch_width,
                              volDepth=int(FLAGS.conv_vol_depth[0]), irDepth=int(FLAGS.conv_ir_depth[0]))
    if (FLAGS.weight_reg_strength is None):
        FLAGS.weight_reg_strength = 0.0

    testX, testY = dataHandler.getTestData()
    x_pl = tf.placeholder(tf.float32, shape=(None, 1, testX.shape[2], testX.shape[3]))
    y_pl = tf.placeholder(tf.float32, shape=(None, testY.shape[1]))
    print("Weight initialization: ", FLAGS.weight_init)

    # if FLAGS.weight_init==WEIGHT_INITIALIZATION_DEFAULT:
    #     w_i=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init](stddev=FLAGS.weight_init_scale)
    # elif FLAGS.weight_init == 'uniform':
    #     FLAGS.weight_init_scale=(-1e-3,1e-3)
    #     print ("Interval :",FLAGS.weight_init_scale)
    #     w_i=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init](-1e-3,1e-3)
    # else :
    #     w_i=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init](FLAGS.weight_init_scale)

    # mlpObj=ConvNet(n_hidden=cnn_hidden_units,n_classes=10,is_training=trainingFlag,activation_fn=ACTIVATION_DICT[FLAGS.activation],\
    # dropout_rate=FLAGS.dropout_rate,weight_initializer=w_i,\
    # weight_regularizer=WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength))
    poolingFlag = tf.placeholder(tf.bool)
    cnn = ConvNet(poolingLayerFlag=poolingFlag)
    pred = cnn.inference(x_pl)
    loss = cnn.loss(pred, y_pl)
    # accuracy=cnn.accuracy(pred,y_pl)
    optimizer = OPTIMIZER_DICT[FLAGS.optimizer](learning_rate=FLAGS.learning_rate)
    opt = optimizer.minimize(loss)

    mergedSummaries = tf.merge_all()
    saver = tf.train.Saver()

    pdb.set_trace()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test', sess.graph)

        for epoch in range(FLAGS.max_steps):
            batch_x, batch_y = dataHandler.getNextBatch()
            _, out, merged_sum = sess.run([opt, loss, mergedSummaries], feed_dict={x_pl: batch_x, y_pl: batch_y})
            if epoch % 100 == 0:
                train_writer.add_summary(merged_sum, epoch)
                train_writer.flush()
                print("====================================================================================")
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.2f}".format(out))
            out, merged_sum = sess.run([loss, mergedSummaries], feed_dict={x_pl: testX, y_pl: testY})
            test_writer.add_summary(merged_sum, epoch)
            test_writer.flush()
            print("Test set:" "loss=", "{:.2f}".format(out), "accuracy=")
            if epoch % FLAGS.checkpoint_freq == 0 and epoch > 0:
                saver.save(sess, FLAGS.checkpoint_dir + '/cnn' + str(epoch) + '.ckpt')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
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


def main(_):
    print_flags()
    initDirectories()
    swo = None
    pdb.set_trace()
    inst.setDataFileName(FLAGS.data_dir)
    if FLAGS.calibrate:
        swo = inst.get_swaptiongen(getIrModel(), FLAGS.currency, FLAGS.irType)
        swo.calibrate_history(start=int(FLAGS.historyStart), end=int(FLAGS.historyEnd))

    if FLAGS.is_train:
        if FLAGS.nn_model == 'cnn':
            trainCNN()
        elif FLAGS.nn_model == 'lstm':
            trainLSTM()
        else:
            raise ValueError("--train_model argument can be lstm or cnn")

    if FLAGS.compare:
        swo.compare_history()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action='store_false', help='Train a model')
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('-bw', '--batch_width', type=int, default=BATCH_WIDTH_DEFAULT,
                        help='Batch width')
    parser.add_argument('-cvd', '--conv_vol_depth', type=str, default=CONVOLUTION_VOL_DEPTH_DEFAULT,
                        help='Comma separated list of number of convolution depth for volatilities in each layer')
    parser.add_argument('-cid', '--conv_ir_depth', type=str, default=CONVOLUTION_IR_DEPTH_DEFAULT,
                        help='Comma separated list of number of convolution depth for interest rate in each layer')
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
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE_DEFAULT, help='Dropout rate.')
    parser.add_argument('--activation', type=str, default=ACTIVATION_DEFAULT,
                        help='Activation function [relu, elu, tanh, sigmoid].')
    parser.add_argument('-o', '--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
    parser.add_argument('-d', '--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('-cf', '--checkpoint_freq', type=str, default=CHECKPOINT_FREQ_DEFAULT,
                        help='How frequently tests will be run')
    parser.add_argument('-tf', '--test_fequency', type=str, default=TEST_FREQUENCY_DEFAULT,
                        help='How frequently tests will be run')
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
    parser.add_argument('--compare', action='store_true', help='Run comparison nn ')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
