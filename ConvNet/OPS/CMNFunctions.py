from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os, sys
from six.moves import cPickle as pickle
from six.moves import range
from collections import defaultdict
import pprint

# Model Packages importA
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.python.framework import ops

from CIFAR_10.DataGenerator import genTrainValidFolds

# class Activations():
def linearActivation(xIN, numInputs, numOutputs, params, scope=None):
    """
    :param x:          Input Data
    :param numInputs:  number of inputs units
    :param numOutputs: number of output units
    :param params:     parameters for linear function Examples {wMean=0.0, wStdev=0.1, wSeed=6432, bSeed=928}
    :param scope:      Scope name (Layer1 or Layer2 depending on what your layer is)
    :return:           A computational graph of the processes of the linear activation
    """
    wMean = params['wMean']
    wStdev = params['wStdev']
    wSeed = params['wSeed']
    bSeed = params['wSeed']

    with tf.variable_scope(scope or "linear-activation"):
        w = tf.get_variable(
                dtype='float32',
                shape=[numInputs, numOutputs],
                initializer=tf.random_normal_initializer(
                        mean=wMean, stddev=wStdev, seed=wSeed),
                name='weight')
        
        b = tf.get_variable(
                dtype='float32',
                shape=[numOutputs],
                initializer=tf.constant_initializer(1.0),
                name='bias')
        
        return tf.matmul(xIN, w) + b


def batchNorm(xIN, numOutputs,
              training_phase,
              mAvg_decay=0.5,
              epsilon=1e-4,
              axes=[0],
              scope=None):
    """
    :param x:           The input after linear activation
    :param numOutputs:  Basically the number of Columns for the input matrix
    :param mAvg_decay:  The moving average decay
                        In many application it is important to keep a moving average of trained variable because
                        sometimes
                        average produce better result than the final value.
                        shadow_variable = decay * shadow_variable + (1 - decay) * variable
                        Here variable contains the new value of the trained variable
                        Here shadow variable contains the moving average till last training point
    :param axes:        For global normalization such as CNN ith [batch, height, width, depth]  pass axes=[0,1,1]
                        For simple batch normalization pass axes=[0]
    :param scope:      The scope name
    :return:           The normed output of from the batch norm layer
    
    : NOTE: The below code can also be written in one line, for example
    tf.nn.batch_normalization(xIN,
                              batch_mean,batch_var,
                              beta,gamma,
                              epsilon)
                              
    But the below equation type code is a better representation for understanding
    
    The parameter beta, gamma and new input xIN are learned via backpropagation
    """
    # First we need to find the Batch mean and standard deviation column wise
    with tf.variable_scope(scope or "batchNorm-Layer"):
        # Initialize parameters for Batch Norm
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOutputs],
                initializer = tf.constant_initializer(0.0),
                name="beta",
                trainable=True
        )
        gamma = tf.get_variable(
                dtype='float32',
                shape=[numOutputs],
                initializer=tf.constant_initializer(1.0),
                name="gamma",
                trainable=True)
        
        # batchMean is an array of Hidden size with mean of each column
        # batchVar is an array of Hidden size with variance of each column
        batchMean, batchVar= tf.nn.moments(xIN, axes, name="moments")
        
        # Initialize the Moving Average model
        ema = tf.train.ExponentialMovingAverage(decay=mAvg_decay)

        # Apply the moving average only for the training Data, not for the crossvalidtion or test data
        def updateMeanVariance():
            # The .apply([list]) function creates the shadow variable for all elements in the list
            # Shadow variables for `Variable` objects are initialized to the variable's initial value.
            maintain_averages_op = ema.apply([batchMean, batchVar])
            # The below takes care of running all the dependency
            with tf.control_dependencies([maintain_averages_op]):
                return tf.identity(batchMean), tf.identity(batchVar)

        
        # The below is analogous to if else condition statement,
        # Basically we don't want to perform the moving average for validation
        # dataset. So we condition that if training_phase is True then we perform
        # mean_var_with_update, Else we print ('11111111')just use the ema (estimated moving average)
        # for both batch_mean and batch_var trained till this point
        mean, var = tf.cond(training_phase,
                            updateMeanVariance,
                            lambda: (ema.average(batchMean), ema.average(batchVar)))
        
        # Normalize the Batch
        bn = (xIN-mean) / tf.sqrt(var + epsilon)

        
        # Scale and shift the normalization, if required
        bnOUT = gamma*bn + beta
        
        return bnOUT, batchMean, batchVar, mean, var
    

def nonLinearActivation(xIN, activation='RELU', scope=None):
    """
    :param x:           The linear activation input/Batch norm input
    :param activation:  The activation unit (RELU/LOGIT)
    :param scope:       Scope name (should be same as the linearActivation scope)
    :return:            The output after applying the non linear activation
    """
    with tf.variable_scope(scope or "nonLinear-activation"):
        if activation=="RELU":
            return tf.nn.relu(xIN)
        elif activation=="LOGIT":
            return tf.sigmoid(xIN)
        




def graphBuilder(xTF, axis, numFeatures, numHid, layerNum, isTraining=True):
    
    print ('1111111111 ',axis)
    print ('')
    print ('2222222222', numFeatures)
    print ('')
    print ('3333333333', numHid)
    print ('')
    print ('4444444444', layerNum)
    # numFeatures = xTF.get_shape().as_list()[1]
    l_layer1 = \
        linearActivation(xIN=xTF,
                         numInputs=numFeatures,
                         numOutputs=numHid,
                         params=dict(
                                wMean=0, wStdev=0.1, wSeed=889, bSeed=716
                         ),
                         scope='Layer%s' % layerNum)

    bn_layer1, batchMean, batchVar, mean, var = \
        batchNorm(xIN=l_layer1,
                  numOutputs=numHid,
                  training_phase=tf.cast(isTraining, tf.bool),
                  mAvg_decay=0.5,
                  epsilon=1e-4,
                  axes=axis,
                  scope='Layer%s' % layerNum)

    nl_layer1 = nonLinearActivation(bn_layer1,
                                    scope='Layer%s' % layerNum)
    
    otherVars = dict(layer1OUT=l_layer1,
                     batchMean=batchMean,
                     batchVar=batchVar,
                     mavgMean=mean,
                     mavgVar=var)
    
    return dict(xTF=xTF,
                linearLayer=l_layer1,
                batchNormLayer=bn_layer1,
                NonLinearLayer=nl_layer1,
                otherVars=otherVars
                )

debug = True
conv = False

if debug:
    ops.reset_default_graph()
    # sess = tf.InteractiveSession()
    
    if conv:
        X = tf.constant(np.array([[[1, 2, 3],
                                   [2, 3, 4]],
                                  [[7, 6, 1],
                                   [1, 1, 1]]
                                  ]))
        axis = [0, 1, 2]
    
        print(X.eval())
    
        batchSize, numFeatures = X.shape
    else:
        
        X = np.array([[1, 2, 3],
                      [2, 3, 4],
                      [7, 6, 1],
                      [1, 1, 1],
                      [0, 4, 2],
                      [1, 1, 5],
                      [1, 3, 2],
                      [2, 2, 2]
                      ], dtype='float32')
        axis = [0]

    unitsPerLayer = [3, 2]
    
    xTF = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='xInputs')
    yTF = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='yInputs')

    #, 64, 128, 128, 64, 64, 3]
    layerGraph = graphBuilder(xTF,
                              axis = axis,
                              numFeatures=unitsPerLayer[0],
                              numHid=unitsPerLayer[1],
                              layerNum=1,
                              isTraining=True)
    
    # sess.run(tf.global_variables_initializer())
    # print([op for op in tf.get_default_graph().get_operations()])

    # print ('11111111')
    # Create the Graph

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print('22222222')
        batches = [X[0:3, :], X[4:7, :]]
        for num, batchData in enumerate(batches):
            batchSize, numFeatures = batchData.shape

            print ('The batchSize is: ', batchSize)
            print ('Total number of features: ', numFeatures)

            print ('')
            print ('Fecthing the outputs from layers')
            print (layerGraph["linearLayer"].eval({xTF: batchData}))
            print ('')
            print (layerGraph["batchNormLayer"].eval({xTF:batchData}))
            # print ('')
            # print (bn_layer1.eval())
            # out = bn_layer1.eval()
            
            print ('layer1OUT = ', layerGraph["otherVars"]["layer1OUT"].eval({xTF:batchData}))
            print ('batchMean = ', layerGraph["otherVars"]["batchMean"].eval({xTF:batchData}))
            print ('batchVar =', layerGraph["otherVars"]["batchVar"].eval({xTF:batchData}))
            print('mavgMean = ', layerGraph["otherVars"]["mavgMean"].eval({xTF:batchData}))
            print("mavgVar =", layerGraph["otherVars"]["mavgVar"].eval({xTF:batchData}))
            # #
            # # Check if the mean = 0 and Var = 1
            # print (np.mean(out, axis=0))
            # print (np.var(out, axis=0))
            #
            # # Check the RELU non-linearity,
            # # check how negative activations are damped to 0
            # print (nl_layer1.eval())
            # print ()

