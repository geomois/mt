from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
import cifar10_siamese_utils

from convnet import *
from siamese import *
from vgg import *

from sklearn.manifold import TSNE
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

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
    optimizer=OPTIMIZER_DICT['adam'](learning_rate=FLAGS.learning_rate)
    train_op=optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None,y_test.shape[1]))
    
    convNet=ConvNet()
    pred=convNet.inference(x_pl)
    loss=convNet.loss(pred,y_pl)
    accuracy=convNet.accuracy(pred,y_pl)
    train_op=train_step(loss)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test',sess.graph)

        for epoch in xrange(FLAGS.max_steps +1):
            batch_x, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
            _,out,acc=sess.run([train_op,loss,accuracy], feed_dict={x_pl: batch_x,y_pl: batch_y})
            if epoch % FLAGS.print_freq == 0:
                # train_writer.add_summary(merged_sum,epoch)
                # train_writer.flush()
                print ("Epoch:", '%05d' % (epoch), "loss=","{:.4f}".format(out),"accuracy=","{:.4f}".format(acc))
            if epoch % FLAGS.eval_freq ==0 and epoch>0:
                avgLoss=0
                avgAcc=0
                count=0
                step=1000
                #Calculating test set in parts, as the whole dataset doesn't fit into the memory 
                for i in xrange(0,x_test.shape[0],step):
                    batch_x=x_test[i:i+step,:]
                    batch_y=y_test[i:i+step]
                    out,acc=sess.run([loss,accuracy], feed_dict={x_pl: batch_x,y_pl: batch_y})
                    avgAcc=avgAcc+acc
                    avgLoss=avgLoss+out
                    count=count+1
                # out,acc=sess.run([loss,accuracy], feed_dict={x_pl: x_test,y_pl:y_test})
                # test_writer.afdd_summary(merged_sum,epoch)
                # test_writer.flush()
                print ("Test set:","accuracy=","{:.4f}".format(avgAcc/count))
            if epoch % FLAGS.checkpoint_freq==0 and epoch>0:
                saver.save(sess,FLAGS.checkpoint_dir+'/linear'+str(epoch)+'.ckpt')

    ########################
    # END OF YOUR CODE    #
    ########################

# def fetch_test_batch():


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_siamese_utils.get_cifar10(FLAGS.data_dir)
    print("Building test dataset..")
    dSet= cifar10_siamese_utils.create_dataset(cifar10.test,num_tuples=500,batch_size=FLAGS.batch_size,fraction_same=0.2)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    x_pl2=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None))
    batch_size_pl=tf.placeholder(tf.float32,shape=(None))

    siamNet=Siamese()
    pred1=siamNet.inference(x_pl)
    pred2=siamNet.inference(x_pl2,reuse=True)
    loss=siamNet.loss(pred1,pred2,y_pl,0.1,batch_size_pl)
    train_op=train_step(loss)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',sess.graph)
        # test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test',sess.graph)
        print ("Training..")
        for epoch in xrange(FLAGS.max_steps +1):
            batch_x1,batch_x2, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
            _,out=sess.run([train_op,loss], feed_dict={x_pl: batch_x1,x_pl2:batch_x2,y_pl: batch_y, batch_size_pl:FLAGS.batch_size})
            if epoch % FLAGS.print_freq == 0:
                # train_writer.add_summary(merged_sum,epoch)
                # train_writer.flush()
                print ("Epoch:", '%05d' % (epoch), "loss=","{:.4f}".format(out))
            if epoch % FLAGS.eval_freq ==0 and epoch>0:
                avgLoss=0
                count=0
                step=10
                for i in xrange(0,len(dSet),step):
                    batch_x1=np.vstack([seq[0] for seq in dSet[i:i+step]])
                    batch_x2=np.vstack([seq[1] for seq in dSet[i:i+step]])
                    batch_y=np.hstack([seq[2] for seq in dSet[i:i+step]])
                    out=sess.run([loss], feed_dict={x_pl: batch_x1,x_pl2:batch_x2,y_pl: batch_y, batch_size_pl:FLAGS.batch_size*step})
                    avgLoss=avgLoss+out[0]
                    count=count+1
                print ("Test set:","loss=","{:.4f}".format(avgLoss/count))
            if epoch % FLAGS.checkpoint_freq==0 and epoch>0:
                saver.save(sess,FLAGS.checkpoint_dir+'/siamese'+str(epoch)+'.ckpt')
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction(model="linear"):
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    batch_size=500 #to be used later for cut training dataset in pieces
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    x_pl2=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None,y_test.shape[1])) 

    print("Building the ",model," model")
    if model=="linear":
        convNet=ConvNet()
        pred=convNet.inference(x_pl)
        loss=convNet.loss(pred,y_pl)
        accuracy=convNet.accuracy(pred,y_pl)
    elif model=="siamese":
        cifar10 = cifar10_siamese_utils.get_cifar10(FLAGS.data_dir)
        # dSet= cifar10_siamese_utils.create_dataset(cifar10.test,num_tuples=50,batch_size=FLAGS.batch_size,fraction_same=0.2)
        y_pl=tf.placeholder(tf.float32,shape=(None)) 
        batch_size_pl=tf.placeholder(tf.float32,shape=(None))
        siamNet=Siamese()
        pred1=siamNet.inference(x_pl)
        pred2=siamNet.inference(x_pl2,reuse=True)
        loss=siamNet.loss(pred1,pred2,y_pl,0.1,batch_size_pl)
    elif model=="vgg":
        retrainFlag=tf.placeholder(tf.bool)
        x_pl=tf.placeholder(tf.float32,shape=(None,None,None,x_test.shape[3]))
        pool5,assign_ops=load_pretrained_VGG16_pool5(x_pl)
        pool5 = tf.cond(retrainFlag, lambda: pool5,lambda:tf.stop_gradient(pool5))
        fcNet=FCNet()
        pred=fcNet.inference(pool5)
        loss=fcNet.loss(pred,y_pl)
        # accuracy=fcNet.accuracy(pred,y_pl)
        # x_test=x_test[:1000]
        # y_test=y_test[:1000]
    

    #Taking intermediate layers

    if model=="siamese":
        l2=tf.get_default_graph().get_tensor_by_name("ConvNet/l2_norm/outNorm:0")
    elif model=="vgg":
        pool=tf.get_default_graph().get_tensor_by_name("vgg/pool5:0")
    else:
        flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/flatten/Flatten/Reshape:0")
        fc1 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc1/activation:0")
        fc2 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc2/activation:0")

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        graph  = tf.get_default_graph()
        check = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir+"/")
        sess.run(tf.initialize_all_variables())

        saver.restore(sess,check.model_checkpoint_path)
        # f=[n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(f)
        # return
#########################################################################save features
        #Save the outcome of intermediate layers
        print("Extracting features")
        if model=="siamese":
            # x_test1=np.vstack([seq[0] for seq in dSet[:]])
            # x_test2=np.vstack([seq[1] for seq in dSet[:]])
            # y_test=np.hstack([seq[2] for seq in dSet[:]])
            # l2Out= sess.run([l2],feed_dict={x_pl: x_test1,x_pl2:x_test2,y_pl:y_test,batch_size_pl:y_test.shape[0]})
            l2Out= sess.run([l2],feed_dict={x_pl: x_test})            
            np.save("./features/"+model+"-l2.npy",l2Out[0])
        elif model=="vgg":
            step=1000
            for i in xrange(0,x_test.shape[0],step):
                batch_x=x_test[i:i+step,:]
                batch_y=y_test[i:i+step]
                pool5Out=sess.run([pool5], feed_dict={x_pl: batch_x,y_pl: batch_y,retrainFlag:False})
                # print("pool ",pool5Out[0].shape)
                np.save("./features/"+model+"_step_"+str(i)+".npy", np.reshape(pool5Out[0],(pool5Out[0].shape[0],pool5Out[0].shape[3])))
                print("extracted")
        else:
            flattenOut,fc1Out,fc2Out = sess.run([flatten,fc1,fc2],feed_dict={x_pl: x_test,y_pl:y_test})
            np.save("./features/"+model+"-flatten.npy", flattenOut)
            np.save("./features/"+model+"-fc1.npy", fc1Out)
            np.save("./features/"+model+"-fc2.npy", fc2Out)  
        # print ("sizes",fc1Out.shape,fc2Out.shape,flattenOut.shape)

        # acc,out = sess.run([accuracy,pred],feed_dict={x_pl: x_test,y_pl:y_test})
        # print ("Test set:","accuracy=","{:.4f}".format(acc))  
        # np.save("./features/out.npy", out)  
#######################################################################################
        print("Classifier started")
        x_train, y_train = cifar10.train.images, cifar10.train.labels
        if model=="siamese":
            clasSiam=OneVsRestClassifier(SVC(kernel='linear'))
        elif model=="vgg":
            clasVgg=OneVsRestClassifier(SVC(kernel='linear'))
        else:
            clasFc1 = OneVsRestClassifier(SVC(kernel='linear'))
            clasFc2 = OneVsRestClassifier(SVC(kernel='linear'))
            clasFlatten = OneVsRestClassifier(SVC(kernel='linear'))
        
        trainXSteps=np.array_split(x_train,x_train.shape[0]/batch_size)
        trainYSteps=np.array_split(y_train,y_train.shape[0]/batch_size)
        # y_all=np.unique(y_train)
        for i in xrange(len(trainXSteps)):
            print("Training classifier step: ",i,"/",len(trainXSteps))
            x_batch=trainXSteps[i]
            y_batch=trainYSteps[i]
            if model=="vgg":
                poolTrain = sess.run([pool],feed_dict={x_pl: x_batch,y_pl:y_batch})
                # clasVgg.fit(np.reshape(poolTrain[0]),y_batch)
                clasVgg.fit(np.reshape(poolTrain[0],(batch_size,512)),y_batch)
            elif model=="linear":
                flattenTrain,fc1Train,fc2Train = sess.run([flatten,fc1, fc2],feed_dict={x_pl: x_batch,y_pl:y_batch})
                clasFlatten.fit(flattenTrain,y_batch)
                clasFc1.fit(fc1Train,y_batch)
                clasFc2.fit(fc2Train,y_batch)
            elif model=="siamese":
                l2Train= sess.run([l2],feed_dict={x_pl: x_batch})
                clasSiam.fit(l2Train[0],y_batch)
            

        print("Predicting test set")
        if model=="siamese":
            outL2=clasSiam.score(l2Out[0],y_test)
            print("Accuracy l2: ",outL2)
        elif model=="vgg":
            step=1000
            outPool=0
            count=0
            for i in xrange(0,x_test.shape[0],step):
                batch_x=x_test[i:i+step,:]
                batch_y=y_test[i:i+step]
                pool5Out=sess.run([pool5], feed_dict={x_pl: batch_x,y_pl: batch_y,retrainFlag:False})
                outPool+=clasVgg.score(np.reshape(pool5Out[0],(pool5Out[0].shape[0],pool5Out[0].shape[3])),batch_y)
                count+=1
            print("Accuracy: ",outPool/count)
        else:
            outFlat=clasFlatten.score(flattenOut,y_test)
            outFc1=clasFc1.score(fc1Out,y_test)
            outFc2=clasFc2.score(fc2Out,y_test)
            print("Accuracy flatten: ",outFlat)
            print("Accuracy fc1: ",outFc1)
            print("Accuracy fc2: ",outFc2)

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
    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        if FLAGS.train_model == 'linear':
            feature_extraction()
        elif FLAGS.train_model == 'siamese':
            feature_extraction(model="siamese")
        elif FLAGS.train_model == 'vgg':
            feature_extraction(model="vgg")
        

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
