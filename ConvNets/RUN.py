
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import cPickle as pickle
from tensorflow.python.framework import ops

from CIFAR_10.DataGenerator import genTrainValidFolds
from Operations.GraphBuilder import convGraphBuilder, nnGraphBuilder, outputToSoftmax
from Operations.Tools import reshape_data, accuracy
from Operations.Preprocessing import Preprocessing
from Operations.CMNFunctions import lossOptimization


def reset_graph():  # Reset the graph
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    

debug = True
conv = True



myNet = dict(imageSize=(32,32),
             numLabels=2,
             numChannels=3,
             convKernel=[(5,5), (5,5)],
             convDepth=[3,64,64],       # The first value of the array should equal to the numChannels
             convStride=[1,1],
             poolKernel=[(2,2), (2,2)],
             poolStride=[1,1],
             fcLayers=[0, 1024, 1024],  # The first value of the array should always be zero because it is updated in
             #  run time
             optimizerParam=dict(optimizer='RMSPROP', learning_rate=0.0001, momentum=0.9),
             batchSize=128,
             epochs = 30)



class GraphComputer():
    def __init__(self, myNet):
        self.myNet = myNet
    

    def computationGraph(self):
        trainTestData = tf.placeholder(dtype=tf.float32,
                                   shape=[None, self.myNet["imageSize"][0],
                                          self.myNet["imageSize"][1],
                                          self.myNet["numChannels"]],
                                   name='xInputs')
    
        trainLabels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.myNet["numLabels"]],
                                     name='yInputs')
        
        runningCount = 1
        
        
        # Convolutions Layers
        layerOutput = trainTestData
        for i in np.arange(2):
            # define what layer you need for one stacked convolution Layer
            layers = ["linear", "batchNorm", "nonLinear", "pool", "regularize"]
            layerOutput, _ = convGraphBuilder(xTF=layerOutput,
                                             convKernelSize=self.myNet["convKernel"][i],
                                             convStride=self.myNet["convStride"][i],
                                             poolKernelSize=self.myNet["poolKernel"][i],
                                             poolStride=self.myNet["poolStride"][i],
                                             inpDepth=self.myNet["convDepth"][i],
                                             outDepth=self.myNet["convDepth"][i + 1],
                                             layerNum=runningCount,
                                             layers=layers,
                                             axis=[0, 1, 2],
                                             isTraining=True)

            runningCount += 1
            
        print ('The shape after convolution layer is : ', layerOutput.get_shape())
        
        # We have to flatten the shape to pass it to the fully connected layer
        # Get the features in flattened fashion
        shapeY, shapeX, depth = layerOutput.get_shape().as_list()[1:4]
        flattenedShape = shapeY * shapeX * depth
        convFeaturesFlattened = tf.reshape(layerOutput, [-1, flattenedShape])
        
        print ('The flattened features of convolutions is: ', convFeaturesFlattened.get_shape())


        # Fully Connected Layers
        self.myNet["fcLayers"][0] = flattenedShape
        layerOutput = convFeaturesFlattened
        for j in np.arange(2):
            k = i+1+j
            # print (self.myNet["fcLayers"][j])
            layers = ["linear", "batchNorm", "nonLinear", "regularize"]
            layerOutput, _ = nnGraphBuilder(xTF=layerOutput,
                                            numInp=self.myNet["fcLayers"][j],
                                            numOut=self.myNet["fcLayers"][j+1],
                                            layerNum = runningCount, layers= layers,
                                            axis=[0], isTraining=True)
            runningCount += 1

        print('The shape after the Fully connected Layer is : ', layerOutput.get_shape())

        
        # Fully connected to Softmax layer
        outState, probLabel = outputToSoftmax(xTF=layerOutput,
                                              numInp=layerOutput.get_shape().as_list()[1],
                                              numOut=self.myNet["numLabels"],
                                              layerNum=runningCount)

        print('The shape of the Tensor after Out to Softmax is : ', probLabel.get_shape())
        
        
        # Loss Function and Optimization
        lossCE, optimizer = lossOptimization(xIN=outState, yIN=trainLabels, optimizerParam = self.myNet[
            "optimizerParam"])
        
        return dict(
                trainTestData=trainTestData,
                trainLabels=trainLabels,
                optimizer=optimizer,
                lossCE=lossCE,
                pred=probLabel,
        )


class SesssionExec():
    
    def __init__(self):
        self.featureDIR = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/featureModels/2-Class/regularFeatures/RGB/batch_data/"
        self.imageSize=32


    def runPreprocessor(self, dataIN, sess):
        preprocessedData = np.ndarray(shape=(dataIN.shape), dtype='float32')
        for numImage in np.arange(dataIN.shape[0]):
            feed_dict = {
                self.preprocessGraphDict['imageIN']:dataIN[numImage,:]
            }
            preprocessedData[numImage,:] = sess.run(self.preprocessGraphDict['imageOUT'],
                                                      feed_dict=feed_dict)
        return preprocessedData


    def trainModel(self, dataIN, labelIN, sess):
        '''
        :param dataIN:      The input data for CIFAR 2 (10000 : 5000 each class)
        :param labelIN:     The input labels to optimize
        :param sess:        Instance for running session
        :return:            Nothing
        
        Here we feed in both the batchData and the baatchLabels and we also run the session
        for loss and optimization because we would want to find gradient and update the
        weights for the training Dataset
        '''
        
        batchSize = myNet["batchSize"]
        numBatches = int(np.ceil(dataIN.shape[0] / batchSize))
        

        for numBatch in np.arange(numBatches):
            batchData = dataIN[numBatch * batchSize: (numBatch + 1) * batchSize]
            batchLabels = labelIN[numBatch * batchSize: (numBatch + 1) * batchSize]
            # print('The shape for Batch Data, Batch Labels is: ', batchData.shape, batchLabels.shape)
            # print('The shape for Batch L is: ', batchData.shape)
            feed_dict = {
                self.compGraphDict['trainTestData']: batchData,
                self.compGraphDict['trainLabels']: batchLabels
            }

            _, loss, trainPred = sess.run([self.compGraphDict['optimizer'],
                                       self.compGraphDict['lossCE'],
                                       self.compGraphDict['pred']],
                                      feed_dict=feed_dict)


            if ((numBatch + 1) % 50 == 0) or ((numBatch + 1) == numBatches):
                trainAcc = accuracy(trainPred, batchLabels)
                print("Fold: " + str(self.foldNUM + 1) +
                      ", Epoch: " + str(self.epoch + 1) +
                      ", Mini Batch: " + str(numBatch + 1) +
                      ", Loss= " + "{:.6f}".format(loss) +
                      ", Training Accuracy= " + "{:.5f}".format(trainAcc))

        return loss, tpred
    
    
    def testModel(self, dataIN, labelIN, sess):
        '''
        :param dataIN:      The input test or validation data
        :param labelIN:     The input labels, (just to compute the accuracy)
        :param sess:        The opened session
        :return:            Nothing
        
        We only run the session for processes resulting only till the "pred" because if we
        call optimization then their would computation for gradients resulting in weight
        change. This would be unwise because we don't want our model to adjust the weight
        for the test data. In other words we dont want out model to learn the test data.
        '''
        batchData = dataIN[numBatch * batchSize: (numBatch + 1) * batchSize]
        batchLabels = labelIN[numBatch * batchSize: (numBatch + 1) * batchSize]
        # print('The shape for Batch Data, Batch Labels is: ', batchData.shape, batchLabels.shape)
        # print('The shape for Batch L is: ', batchData.shape)
        feed_dict = {
            self.compGraphDict['trainData']: dataIN
        }
    
        testPred = sess.run([self.compGraphDict['pred']],
                                  feed_dict=feed_dict)
    
        
        testAcc = accuracy(testPred, batchLabels)
        print("Fold: " + str(self.foldNUM + 1) +
              ", Epoch: " + str(self.epoch + 1) +
              ", Mini Batch: " + str(numBatch + 1) +
              ", Loss= " + "{:.6f}".format(loss) +
              ", Training Accuracy= " + "{:.5f}".format(testAcc))


        
    def execute(self):
        meanValidAcc = 0
        for foldNUM, (trainDataIN, trainLabelsIN,
                      validDataIN, validLabelsIN, labelDict
                      ) in enumerate(
                genTrainValidFolds(self.featureDIR, oneHot=True)):
            
            self.foldNUM = foldNUM
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
            self.preprocessGraphDict = Preprocessing().preprocessImageGraph(
                                                            imageSize=myNet["imageSize"],
                                                            numChannels=myNet["numChannels"])
            
            self.compGraphDict = computationGraph(myNet).trainGraph()
            
            
            # We would like to calculate the accuracy after each epoch
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch in range(myNet["epochs"]):
                    self.epoch = epoch
                    # print([op for op in tf.get_default_graph().get_operations()])
                    preprocessedTrainData = self.runPreprocessor(dataIN=trainDataIN, sess=sess)
                    
                    print ('################## ', preprocessedTrainData.shape)
    
                    loss, tpred = self.trainModel(dataIN=trainDataIN,
                                                  labelIN=trainLabelsIN,
                                                  sess=sess)
                    
            break
                
            # Now we create the training Graph
            

            

            
SesssionExec().execute()
