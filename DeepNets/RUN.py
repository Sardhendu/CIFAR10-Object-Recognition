
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from CIFAR_10.DataGenerator import genTrainValidFolds
from Operations.GraphBuilder import convGraphBuilder, nnGraphBuilder
from Operations.Tools import reshape_data, accuracy
from Operations.Preprocessing import Preprocessing


def reset_graph():  # Reset the graph
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    

debug = True
conv = True



myNet = dict(imageSize=(32,32),
             numLabels=2,
             numChannels=3,
             hiddenLayers=[1024,1024],
             convKernel=[(5,5), (5,5)],
             convDepth=[3,64,64],
             convStride=[1,1],
             poolKernel=[(2,2), (2,2)],
             poolStride=[1,1])



class GraphComputer():
    def __init__(self, myNet):
        self.myNet = myNet
    

    def trainGraph(self):
        trainData = tf.placeholder(dtype=tf.float32,
                                   shape=[None, self.myNet["imageSize"][0],
                                          self.myNet["imageSize"][1],
                                          self.myNet["numChannels"]],
                                   name='xInputs')
    
        trainLabels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.myNet["numLabels"]],
                                     name='yInputs')
        
        # First we Preprocess the input
        layerOutput = trainData
        for i in np.arange(2):
            # define what layer you need for one stacked convolution Layer
            layers = ["linear", "batchNorm", "nonLinear", "pool"]
            layerOutput, _ = convGraphBuilder(xTF=layerOutput,
                                             axis=[0, 1, 2],
                                             convKernelSize=self.myNet["convKernel"][i],
                                             convStride=self.myNet["convStride"][i],
                                             poolKernelSize=self.myNet["poolKernel"][i],
                                             poolStride=self.myNet["poolStride"][i],
                                             inpDepth=self.myNet["convDepth"][i],
                                             outDepth=self.myNet["convDepth"][i + 1],
                                             layerNum=i + 1,
                                             layers=layers,
                                             isTraining=True)

           
        print ('The shape is : ', layerOutput.get_shape())
        # We have to flatten the shape to pass it to the fully connected layer
        
        
        # for i in np.arange(2):
        #     layers
    
        return dict(
                trainData=trainData,
                trainLabels=trainLabels,
                layerOutput = layerOutput
                # optimizer=optimizer,
                # lossCE=lossCE,
                # trainPred=outputState,
                # wghtNew=self.weights,
                # poolShape=poolShape
        )
    
    
    
    
class SesssionExec():
    
    def __init__(self):
        self.featureDIR = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/featureModels/2-Class/regularFeatures/RGB/batch_data/"
        self.imageSize=32


    # def runPreprocessor(self, dataIN, sess):
    #     dataOut = np.ndarray(shape=(dataIN.shape[0],dataIN.shape[1],dataIN.shape[2],dataIN.shape[3]), dtype='float32')
    #     for numImage in np.arange(dataIN.shape[0]):
    #         feed_dict = {
    #             self.preprocessGraphDict['imageIN']:dataIN[numImage,:]
    #         }
    #         dataOut[numImage,:] = sess.run(self.preprocessGraphDict['distorted_image'],
    #                                                   feed_dict=feed_dict)
    #     return dataOut
    
    
    
        
        
    def execute(self):
        meanValidAcc = 0
        for foldNUM, (trainDataIN, trainLabelsIN, validDataIN, validLabelsIN, labelDict) in enumerate(
                genTrainValidFolds(self.featureDIR, oneHot=True)):
            print('')
            print('##########################################################################################')
            trainDataIN, _ = reshape_data(trainDataIN,
                                          imageSize=myNet["imageSize"][0],
                                          numChannels=myNet["numChannels"])
            
            validDataIN, _ = reshape_data(validDataIN,
                                          imageSize=myNet["imageSize"][0],
                                          numChannels=myNet["numChannels"])
            print('')
            print('Validation Data and Labels shape: ', validDataIN.shape, validLabelsIN.shape)
            print('Training Data and Labels shape: ', trainDataIN.shape, trainLabelsIN.shape)
            print('The Label Dictionary is given as: ', labelDict)
            print('')
            
            # Step 1: First we create the Pre-processing Graph:
            preprocessDict = Preprocessing().preprocessImageGraph(
                    imageSize=myNet["imageSize"],
                    numChannels=myNet["numChannels"])
            
            trainDict = GraphComputer(myNet).trainGraph()
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                print([op for op in tf.get_default_graph().get_operations()])
                
            break
                
            # Now we create the training Graph
            

            

            
SesssionExec().execute()
        

        #
        #     self.foldNUM = foldNUM
        #
        #     reset_graph()
        #
        #     # Create a object encapsulating the graph lineage
        #     objCNN = BuildConvNet(params)
        #     self.preprocessGraphDict = objCNN.preprocessImageGraph()
        #     self.trainGraphDict = objCNN.trainGraph()
        #     self.validGraphDict = objCNN.validGraph()
        #
        #
        #
        #     axis = [0, 1, 2]
        #     batchSize, imageY, imageX, numChannels = X.shape
        #     numLabels = 2
        #     print('The input Data shape is: ', X.shape)
        #
        #     xTF = tf.placeholder(dtype=tf.float32,
        #                          shape=[None, imageY, imageX, numChannels],
        #                          name='xInputs')
        #
        #     yTF = tf.placeholder(dtype=tf.float32,
        #                          shape=[None, numLabels],
        #                          name='yInputs')
        #
        #     # kernel1X , kernel1Y = 3,3
        #     kernel1D = 2
        #     layerGraph = convGraphBuilder(xTF, axis,
        #                                   kernelSize=(3, 3),
        #                                   inpDepth=numChannels,
        #                                   outDepth=kernel1D,
        #                                   layerNum=1,
        #                                   isTraining=True)
        #
        #
        # else:
        #     X = np.array([[1, 2, 3],
        #                   [2, 3, 4],
        #                   [7, 6, 1],
        #                   [1, 1, 1],
        #                   [0, 4, 2],
        #                   [1, 1, 5],
        #                   [1, 3, 2],
        #                   [2, 2, 2]
        #                   ], dtype='float32')
        #     axis = [0]
        #
        #     xTF = tf.placeholder(dtype=tf.float32,
        #                          shape=[None, 3],
        #                          name='xInputs')
        #     yTF = tf.placeholder(dtype=tf.float32,
        #                          shape=[None, 3],
        #                          name='yInputs')
        #
        #     unitsPerLayer = [2, 3]
        #
        #     # , 64, 128, 128, 64, 64, 3]
        #     layerGraph = nnGraphBuilder(xTF,
        #                                 axis=axis,
        #                                 numIN=unitsPerLayer[0],
        #                                 numOut=unitsPerLayer[1],
        #                                 layerNum=1,
        #                                 isTraining=True)
        #
        # # sess.run(tf.global_variables_initializer())
        # # print([op for op in tf.get_default_graph().get_operations()])
        #
        # # print ('11111111')
        # # Create the Graph
        #
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     # print('22222222')
        #     batches = [X[0:4, :]]  # , X[4:7, :]]
        #
        #     for num, batchData in enumerate(batches):
        #         print(batchData.shape, '\n', batchData)
        #         print('')
        #         batchSize, imageY, imageX, numFeatures = batchData.shape
        #
        #         print('The batchSize is: ', batchSize)
        #         print('Total number of features: ', numFeatures)
        #
        #         print('')
        #         print('Fetching the outputs from layers')
        #         print(layerGraph["convLayer"].eval({xTF: batchData}))
        #         print('')
        #         print(layerGraph["batchNormLayer"].eval({xTF: batchData}))
        #         print ('')
        #         print (layerGraph["nonLinearLayer"].eval({xTF: batchData}))
        #         # print (bn_layer1.eval())
        #         # # out = bn_layer1.eval()
        #         #
        #         # print ('layer1OUT = ', layerGraph["otherVars"]["layer1OUT"].eval({xTF :batchData}))
        #         # print ('batchMean = ', layerGraph["otherVars"]["batchMean"].eval({xTF :batchData}))
        #         # print ('batchVar =', layerGraph["otherVars"]["batchVar"].eval({xTF :batchData}))
        #         # print('mavgMean = ', layerGraph["otherVars"]["mavgMean"].eval({xTF :batchData}))
        #         # print("mavgVar =", layerGraph["otherVars"]["mavgVar"].eval({xTF :batchData}))
        #         # #
        #         # # Check if the mean = 0 and Var = 1
        #         # print (np.mean(out, axis=0))
        #         # print (np.var(out, axis=0))
        #         #
        #         # # Check the RELU non-linearity,
        #         # # check how negative activations are damped to 0
        #         # print (nl_layer1.eval())
        #         # print ()
        #
