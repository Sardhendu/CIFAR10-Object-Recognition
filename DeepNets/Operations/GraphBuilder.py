from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from Operations.CMNFunctions import convLinearActivation, batchNorm, \
    nonLinearActivation, poolLayer, dropout, linearActivation, softmaxActivation




def convGraphBuilder(xTF, convKernel, convStride, inpDepth, outDepth,
                     poolKernel, poolStride, dropoutParam, isTraining, layers, layerNum,
                     axis=[0,1,2]):
    padding = "SAME"
    poolType = "MAX"
    nonLinearAct = "RELU"
    mAvg_decay = 0.5
    epsilon = 1e-4
    scope = 'Layer%s' % str(layerNum)
    
    individual_layer_dict = dict()
    other_vars = dict()
    
    if "linear" in layers:
        layerActivation = convLinearActivation(
                xIN=xTF,
                convKernelSize=convKernel,
                convStride=convStride,
                inpDepth=inpDepth,
                outDepth=outDepth,
                padding=padding,
                params=dict(
                        wMean=0, wStdev=0.1,
                        wSeed=231, bSeed=443),
                scope=scope)

        individual_layer_dict.update(
                dict(convLayer=layerActivation)
        )
    

    if "batchNorm" in layers:
        layerActivation, batchMean, batchVar, mean, var = \
            batchNorm(xIN=layerActivation,
                      numOut=outDepth,
                      training_phase=isTraining,
                      mAvg_decay=mAvg_decay,
                      epsilon=epsilon,
                      axes=axis,
                      scope=scope)

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
                                activation=nonLinearAct,
                                scope=scope)

        individual_layer_dict.update(
                dict(nonLinearLayer=layerActivation)
        )

    
    
    if "pool" in layers:
        layerActivation = poolLayer(xIN=layerActivation,
                                    poolKernelSize=poolKernel,
                                    poolStride=poolStride,
                                    padding=padding,
                                    poolType=poolType,
                                    scope=scope
                                    )


    if "dropout" in layers:
        layerActivation = dropout(
                xIN=layerActivation,
                decayParam=dropoutParam
        )
        
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)





def nnGraphBuilder(xTF, numInp, numOut, dropoutParam, isTraining, layers,
                   layerNum, axis=[0]):
    

    nonLinearAct = "RELU"
    mAvg_decay = 0.5
    epsilon = 1e-4
    scope = 'Layer%s' % str(layerNum)
    
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
                             scope=scope)

        individual_layer_dict.update(
                dict(convLayer=layerActivation)
        )


    if "batchNorm" in layers:
        layerActivation, batchMean, batchVar, mean, var = \
            batchNorm(xIN=layerActivation,
                      numOut=numOut,
                      training_phase=tf.cast(isTraining, tf.bool),
                      mAvg_decay=mAvg_decay,
                      epsilon=epsilon,
                      axes=axis,
                      scope=scope)

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
                activation=nonLinearAct,
                scope=scope)

        individual_layer_dict.update(
                dict(nonLinearLayer=layerActivation)
        )


        
    if "dropout" in layers:
        layerActivation = dropout(
                xIN=layerActivation,
                decayParam=dropoutParam
        )
    
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)




def outputToSoftmax(xTF, numInp, numOut,
                    layerNum):
    outState, probLabel = \
                softmaxActivation(xIN=xTF,
                                 numInp=numInp,
                                 numOut=numOut,
                                 params=dict(
                                         wMean=0, wStdev=0.1, wSeed=889, bSeed=716
                                 ),
                                 scope='Layer%s' % layerNum)
    
    return outState, probLabel