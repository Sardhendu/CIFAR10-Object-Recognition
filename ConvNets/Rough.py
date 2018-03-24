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
    wMean = params['wMean']
    wStdev = params['wStdev']
    wSeed = params['wSeed']
    bSeed = params['wSeed']
    

    w = tf.cast(np.random.normal(size=[numInputs, numOutputs]),
                dtype=tf.float32)
        
    b = tf.cast(np.full((numOutputs), 1.0), dtype=tf.float32)
    
    linearOut = tf.matmul(xIN, w) + b
    
    print ('The Linear output looks like, %s \n'%(linearOut.get_shape()),
           linearOut.eval())
    return linearOut


def batchNorm(xIN,
              numOutputs,
              training_phase,
              mAvg_decay=0.5,
              epsilon=1e-4,
              axes=[0],
              scope=None):
    
    # First we need to find the Batch mean and standard deviation column wise
    with tf.variable_scope(scope or "batchNorm-Layer"):
        # Initialize parameters for Batch Norm
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOutputs],
                initializer=tf.constant_initializer(0.0),
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
        batchMean, batchVar = tf.nn.moments(xIN, axes, name="moments")
        print ("Batch Mean and variance are: \n",
               batchMean.eval(),
               batchVar.eval())
        
        # Initialize the Moving Average model
        ema = tf.train.ExponentialMovingAverage(decay=mAvg_decay)
        # print("ema are: ", ema.eval())
        
        # Apply the moving average only for the training Data, not for the crossvalidtion or test data
        def updateMeanVariance():
            # The .apply([list]) function creates the shadow variable for all elements in the list
            # Shadow variables for `Variable` objects are initialized to the variable's initial value.
            print ('YUPYUOYUPYUOYPYIYUYUPYUO')
            maintain_averages_op = ema.apply([tf.cast(batchMean, tf.float32), tf.cast(batchVar, tf.float32)])
            # print ("maintain_averages_op \n",
            #        maintain_averages_op)
            # The below takes care of running all the dependency
            with tf.control_dependencies([maintain_averages_op]):
                return tf.identity(batchMean), tf.identity(batchVar)
        
        # The below is analogous to if else condition statement,
        # Basically we don't want to perform the moving average for validation
        # dataset. So we condition that if training_phase is True then we perform
        # mean_var_with_update, Else we print ('11111111')just use the ema (estimated moving average)
        # for both batch_mean and batch_var trained till this point
        print ('Executing for Training Phase ', training_phase.eval())
        mean, var = tf.cond(training_phase,
                            updateMeanVariance,
                            lambda: (ema.average(batchMean), ema.average(batchVar)))
        
        print ('The new mean and variance after moving average decay is: \n',
               mean.eval(), var.eval())
        
        # Normalize the Batch
        bn = (xIN - mean) / tf.sqrt(var + epsilon)
        
        # Scale and shift the normalization, if required
        bnOUT = gamma * bn + beta
        
        return bnOUT


def nonLinearActivation(xIN, activation='RELU', scope=None):
    """
    :param x:           The linear activation input/Batch norm input
    :param activation:  The activation unit (RELU/LOGIT)
    :param scope:       Scope name (should be same as the linearActivation scope)
    :return:            The output after applying the non linear activation
    """
    with tf.variable_scope(scope or "nonLinear-activation"):
        if activation == "RELU":
            return tf.nn.relu(xIN)
        elif activation == "LOGIT":
            return tf.sigmoid(xIN)


def graphBuilder(xTF, axis, numFeatures, numHid, layerNum, isTraining=True):
    # print('1111111111 ', axis)
    # print('')
    # print('2222222222', numFeatures)
    # print('')
    # print('3333333333', numHid)
    # print('')
    # print('4444444444', layerNum)
    # numFeatures = xTF.get_shape().as_list()[1]
    linearACTout = \
        linearActivation(xIN=xTF,
                         numInputs=numFeatures,
                         numOutputs=numHid,
                         params=dict(
                                 wMean=0, wStdev=0.1, wSeed=889, bSeed=716
                         ),
                         scope='Layer%s' % layerNum)
    print ("Properly executed the linear activation ..........")
    
    bn_layer1 = \
        batchNorm(xIN=linearACTout,
                  numOutputs=numHid,
                  training_phase=tf.cast(isTraining, tf.bool),
                  mAvg_decay=0.5,
                  epsilon=1e-4,
                  axes=axis,
                  scope='Layer%s' % layerNum)

    # nl_layer1 = nonLinearActivation(bn_layer1,
    #                                 scope='Layer%s' % layerNum)
    #
    #
    #
    # return dict(xTF=xTF,
    #             linearLayer=l_layer1,
    #             batchNormLayer=bn_layer1,
    #             NonLinearLayer=nl_layer1,
    #             )


debug = False
conv = False

if debug:
    ops.reset_default_graph()
    sess = tf.InteractiveSession()
    
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

    batches = [X[0:3, :], X[4:7, :]]
    for num, batchData in enumerate(batches):
        xTF = tf.cast(batchData, dtype=tf.float32)
        yTF = tf.cast(batchData, dtype=tf.float32)
        
        batchSize, numFeatures = batchData.shape
        
        # , 64, 128, 128, 64, 64, 3]
        graphBuilder(xTF,
                      axis=axis,
                      numFeatures=unitsPerLayer[0],
                      numHid=unitsPerLayer[1],
                      layerNum=num,
                      isTraining=True)
    
