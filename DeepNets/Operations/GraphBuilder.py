from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from Operations.CMNFunctions import convLinearActivation, batchNorm, \
    nonLinearActivation, poolLayer, linearActivation, outSoftmaxActivation


def convGraphBuilder(xTF, convKernelSize, convStride,
                     poolKernelSize, poolStride,
                     inpDepth, outDepth, layerNum, layers,
                     axis=[0,1,2], isTraining=True):
    
    individual_layer_dict = dict()
    other_vars = dict()
    
    if "linear" in layers:
        layerActivation = convLinearActivation(
                xIN=xTF,
                convKernelSize=convKernelSize,
                convStride=convStride,
                inpDepth=inpDepth,
                outDepth=outDepth,
                padding='SAME',
                params=dict(
                        wMean=0, wStdev=0.1,
                        wSeed=231, bSeed=443),
                scope='Layer%s' % str(layerNum))

        individual_layer_dict.update(
                dict(convLayer=layerActivation)
        )
    
    if "batchNorm" in layers:
        layerActivation, batchMean, batchVar, mean, var = \
            batchNorm(xIN=layerActivation,
                      numOut=outDepth,
                      training_phase=tf.cast(isTraining, tf.bool),
                      mAvg_decay=0.5,
                      epsilon=1e-4,
                      axes=axis,
                      scope='Layer%s' % str(layerNum))

        individual_layer_dict.update(
                dict(batchNormLayer=layerActivation)
        )

        other_vars.update(
                dict(batchMean=batchMean,
                     batchVar = batchVar,
                     mavgMean=mean,
                     mavgVar=var
                     )
        )
        
    if "nonLinear" in layers:
        layerActivation = nonLinearActivation(
                                xIN=layerActivation,
                                activation='RELU',
                                scope="Layer%s" % str(layerNum))

        individual_layer_dict.update(
                dict(nonLinearLayer=layerActivation)
        )
        
    if "pool" in layers:
        layerActivation = poolLayer(xIN=layerActivation,
                                    poolKernelSize=poolKernelSize,
                                    poolStride=poolStride,
                                    padding="SAME",
                                    poolType='MAX',
                                    scope="Layer%s" % str(layerNum))
    
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)





def nnGraphBuilder(xTF, numInp, numOut,
                   layerNum, layers, axis=[0], isTraining=True):
    
    individual_layer_dict = dict()
    other_vars = dict()
    
    if "linear" in layers:
        layerActivation = \
            linearActivation(xIN=xTF,
                             numInp=numInp,
                             numOut=numOut,
                             params=dict(
                                     wMean=0, wStdev=0.1, wSeed=889, bSeed=716
                             ),
                             scope='Layer%s' % layerNum)

        individual_layer_dict.update(
                dict(convLayer=layerActivation)
        )
    
    if "batchNorm" in layers:
        layerActivation, batchMean, batchVar, mean, var = \
            batchNorm(xIN=layerActivation,
                      numOut=numOut,
                      training_phase=tf.cast(isTraining, tf.bool),
                      mAvg_decay=0.5,
                      epsilon=1e-4,
                      axes=axis,
                      scope='Layer%s' % str(layerNum))

        individual_layer_dict.update(
                dict(batchNormLayer=layerActivation)
        )

        other_vars.update(
                dict(batchMean=batchMean,
                     batchVar=batchVar,
                     mavgMean=mean,
                     mavgVar=var
                     )
        )
        
    if "nonLinear" in layers:
        layerActivation = nonLinearActivation(
                xIN=layerActivation,
                activation='RELU',
                scope="Layer%s" % str(layerNum))

        individual_layer_dict.update(
                dict(nonLinearLayer=layerActivation)
        )
    
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)




def outputToSoftmax(xTF, numInp, numOut,
                    layerNum):
    outState, softmax = \
            outSoftmaxActivation(xIN=xTF,
                                 numInp=numInp,
                                 numOut=numOut,
                                 params=dict(
                                         wMean=0, wStdev=0.1, wSeed=889, bSeed=716
                                 ),
                                 scope='Layer%s' % layerNum)
    
    return outState, softmax