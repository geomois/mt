from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
from vgg import *
import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'adam'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    optimizer=OPTIMIZER_DICT[OPTIMIZER_DEFAULT](learning_rate=FLAGS.learning_rate)
    train_op=optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,None,None,x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None,y_test.shape[1]))
    retrainFlag=tf.placeholder(tf.bool)

    pool5,assign_ops=load_pretrained_VGG16_pool5(x_pl)
    pool5 = tf.cond(retrainFlag, lambda: pool5,lambda:tf.stop_gradient(pool5))
    fcNet=FCNet()
    pred=fcNet.inference(pool5)
    loss=fcNet.loss(pred,y_pl)
    accuracy=fcNet.accuracy(pred,y_pl)
    train_op=train_step(loss)
    saver=tf.train.Saver()
    merged=tf.merge_all_summaries()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for op in assign_ops:
            sess.run(op)
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test',sess.graph)
        print("Starting training")
        for epoch in range(FLAGS.max_steps +1):
            batch_x, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
            if FLAGS.refine_after_k >= epoch:
                afterKBool=False
            else:
                afterKBool=True
            # print("kBool",afterKBool)
            _,out,acc,merged_sum=sess.run([train_op,loss,accuracy,merged], feed_dict={x_pl: batch_x,y_pl: batch_y,retrainFlag:afterKBool})
            if epoch % FLAGS.print_freq == 0:
                train_writer.add_summary(merged_sum,epoch)
                train_writer.flush()
                print ("Epoch:", '%05d' % (epoch), "loss=","{:.4f}".format(out),"accuracy=","{:.4f}".format(acc))
            if epoch % FLAGS.eval_freq ==0 and epoch>0:
                avgLoss=0
                avgAcc=0
                count=0
                step=1000
                merged_sum_test=None
                #Calculating test set in parts, as the whole dataset doesn't fit into the memoryview
                for i in range(0,x_test.shape[0],step):
                    batch_x=x_test[i:i+step,:]
                    batch_y=y_test[i:i+step]
                    out,acc,merged_sum_test=sess.run([loss,accuracy,merged], feed_dict={x_pl: batch_x,y_pl: batch_y,retrainFlag:False})
                    avgAcc=avgAcc+acc
                    avgLoss=avgLoss+out
                    count=count+1
                test_writer.add_summary(merged_sum_test,epoch)
                test_writer.flush()
                print ("Test set:","accuracy=","{:.4f}".format(avgAcc/count))
            if epoch % FLAGS.checkpoint_freq==0 and epoch>0:
                saver.save(sess,FLAGS.checkpoint_dir+'/vgg'+str(epoch)+"_ref"+str(FLAGS.refine_after_k)+'.ckpt')
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
