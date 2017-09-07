from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from Operations.CMNFunctions import convLinearActivation, batchNorm, \
    nonLinearActivation, poolLayer, dropout, linearActivation, softmaxActivation


padding = "SAME"
poolType = "MAX"
nonLinearAct = "RELU"
mAvg_decay = 0.5
epsilon = 1e-4


def convGraphBuilder(xTF, convParams, poolParams, dropoutParams,
                     isTraining, layers, layerNum, axis=[0,1,2]):
   
    scope = 'ConvLayer%s' % str(layerNum)
    
    individual_layer_dict = dict()
    other_vars = dict()

    kernelY, kernelX, inpDepth, outDepth = convParams["shape"]
    
    if "linear" in layers:
        layerActivation = convLinearActivation(
                xIN=xTF,
                convShape=convParams["shape"],
                stride=convParams["stride"],
                padding=padding,
                wgthMean=convParams["wghtMean"],
                wghtStddev=convParams["wghtStddev"],
                bias=convParams["bias"],
                seed=convParams["seed"],
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
                                    poolShape=poolParams["shape"],
                                    poolStride=poolParams["stride"],
                                    padding=padding,
                                    poolType=poolType,
                                    scope=scope
                                    )


    if "dropout" in layers:
        layerActivation = dropout(
                xIN=layerActivation,
                keepProb=dropoutParams["keepProb"],
                seed = dropoutParams["seed"]
        )
        
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)





def nnGraphBuilder(xTF, linearParams, dropoutParams, isTraining, layers,
                   layerNum, axis=[0]):

    scope = 'Layer%s' % str(layerNum)
    
    individual_layer_dict = dict()
    other_vars = dict()

    numInp, numOut = linearParams["shape"]
    if "linear" in layers:
        layerActivation = \
            linearActivation(xIN=xTF,
                             inpOutShape=linearParams["shape"],
                             wghtMean=linearParams["wghtMean"],
                             wghtStddev=linearParams["wghtStddev"],
                             bias=linearParams["bias"],
                             seed=linearParams["seed"],
                             scope=scope)

        individual_layer_dict.update(
                dict(convLayer=layerActivation)
        )


    if "batchNorm" in layers:
        layerActivation, batchMean, batchVar, mean, var = \
            batchNorm(xIN=layerActivation,
                      numOut=numOut,
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
                keepProb=dropoutParams["keepProb"],
                seed = dropoutParams["seed"]
        )
    
    return layerActivation, dict(xTF=xTF,individual_layer_dict=individual_layer_dict,other_vars=other_vars)




def softmaxLayer(xTF, softmaxParams):
    scope = "SoftmaxLayer"

    outputState = \
        linearActivation(xIN=xTF,
                         inpOutShape=softmaxParams["shape"],
                         wghtMean=softmaxParams["wghtMean"],
                         wghtStddev=softmaxParams["wghtStddev"],
                         bias=softmaxParams["bias"],
                         seed=softmaxParams["seed"],
                         scope=scope)

    probLabel = softmaxActivation(outputState, scope)
    # individual_layer_dict.update(
    #
    #         outState, probLabel = \
    #             softmaxActivation(xIN=xTF,
    #                              numInp=numInp,
    #                              numOut=numOut,
    #                              params=dict(
    #                                      wMean=0, wStdev=0.1, wSeed=889, bSeed=716
    #                              ),
    #                              scope='Layer%s' % layerNum)
    
    return outputState, probLabel




def summaryBuilder(sess, outFilePath):
    mergedSummary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(outFilePath)
    writer.add_graph(sess.graph)
    return mergedSummary, writer