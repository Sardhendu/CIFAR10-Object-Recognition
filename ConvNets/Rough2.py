from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from CIFAR_10.DataGenerator import genTrainValidFolds
from Operations.GraphBuilder import convGraphBuilder, nnGraphBuilder

debug = True
conv = True

if debug:
    ops.reset_default_graph()
    # sess = tf.InteractiveSession()
    
    if conv:
        X = np.array([[[1, 2, 3],
                       [2, 3, 4],
                       [3, 3, 3],
                       [4, 1, 2]],
                      [[7, 6, 1],
                       [1, 1, 1],
                       [2, 3, 4],
                       [1, 1, 2]],
                      [[6, 6, 5],
                       [3, 2, 4],
                       [2, 3, 4],
                       [1, 1, 3]],
                      [[2, 1, 2],
                       [1, 3, 3],
                       [4, 4, 2],
                       [5, 4, 3]]
                      ])
        print('1212121212121 ', X.shape)
        X = np.array([X], dtype='float32')
        print('3232323232323 ', X.shape)
        
        axis = [0, 1, 2]
        batchSize, imageY, imageX, numChannels = X.shape
        numLabels = 2
        print('The input Data shape is: ', X.shape)
        
        xTF = tf.placeholder(dtype=tf.float32,
                             shape=[None, imageY, imageX, numChannels],
                             name='xInputs')
        
        yTF = tf.placeholder(dtype=tf.float32,
                             shape=[None, numLabels],
                             name='yInputs')
        
        # kernel1X , kernel1Y = 3,3
        kernel1D = 2
        layerGraph = convGraphBuilder(xTF, axis,
                                      kernelSize=(3, 3),
                                      inpDepth=numChannels,
                                      outDepth=kernel1D,
                                      layerNum=1,
                                      isTraining=True)
    
    
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
        
        xTF = tf.placeholder(dtype=tf.float32,
                             shape=[None, 3],
                             name='xInputs')
        yTF = tf.placeholder(dtype=tf.float32,
                             shape=[None, 3],
                             name='yInputs')
        
        unitsPerLayer = [2, 3]
        
        # , 64, 128, 128, 64, 64, 3]
        layerGraph = nnGraphBuilder(xTF,
                                    axis=axis,
                                    numIN=unitsPerLayer[0],
                                    numOut=unitsPerLayer[1],
                                    layerNum=1,
                                    isTraining=True)
    
    # sess.run(tf.global_variables_initializer())
    # print([op for op in tf.get_default_graph().get_operations()])
    
    # print ('11111111')
    # Create the Graph
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print('22222222')
        batches = [X[0:4, :]]  # , X[4:7, :]]
        
        for num, batchData in enumerate(batches):
            print(batchData.shape, '\n', batchData)
            print('')
            batchSize, imageY, imageX, numFeatures = batchData.shape
            
            print('The batchSize is: ', batchSize)
            print('Total number of features: ', numFeatures)
            
            print('')
            print('Fetching the outputs from layers')
            print(layerGraph["convLayer"].eval({xTF: batchData}))
            print('')
            print(layerGraph["batchNormLayer"].eval({xTF: batchData}))
            print('')
            print(layerGraph["nonLinearLayer"].eval({xTF: batchData}))
            # print (bn_layer1.eval())
            # # out = bn_layer1.eval()
            #
            # print ('layer1OUT = ', layerGraph["otherVars"]["layer1OUT"].eval({xTF :batchData}))
            # print ('batchMean = ', layerGraph["otherVars"]["batchMean"].eval({xTF :batchData}))
            # print ('batchVar =', layerGraph["otherVars"]["batchVar"].eval({xTF :batchData}))
            # print('mavgMean = ', layerGraph["otherVars"]["mavgMean"].eval({xTF :batchData}))
            # print("mavgVar =", layerGraph["otherVars"]["mavgVar"].eval({xTF :batchData}))
            # #
            # # Check if the mean = 0 and Var = 1
            # print (np.mean(out, axis=0))
            # print (np.var(out, axis=0))
            #
            # # Check the RELU non-linearity,
            # # check how negative activations are damped to 0
            # print (nl_layer1.eval())
            # print ()

