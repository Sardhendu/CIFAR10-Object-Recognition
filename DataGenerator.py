"""
Author: Sardhendu Mishra

Code Info: Central to this module is "genTrainValidFolds". This function generates batch data of n foldes and provide the caller function will n-1 fold for training and the nth fold for validation.

"""


import os, sys,glob
import numpy as np
import pickle

# from Tools import GlobalVariables

# stdBatchPath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/featureSTD/batchPath/'

def readFiles(dirIN):
    with open(dirIN, 'rb') as f:
        data = pickle.load(f)
    return data['batchData'], data['batchLabels'], data['labelDict']      

def reshape_data(dataset, labels, num_features, numLabels, sample_size=None):
    if sample_size:
        dataset = dataset[:sample_size].reshape(sample_size, num_features) # To reshape the  
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(numLabels) == labels[:,None]).astype(np.float32)
    else:
        dataset = dataset.reshape(len(dataset), num_features) # To reshape the  
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(numLabels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def genBatchIterator(self,batch_dir, batch_filename_arr):
    for no, batch_file in enumerate(batch_filename_arr):
        trnBatchData, trnBatchLabels = readFiles(os.path.join(batch_dir,batch_file))
        yield trnBatchData, trnBatchLabels


# def genTrainValidFolds(filePath):
# 	# stdBatchPath  = GlobalVariables()
# 	batches = np.array([files for files in os.listdir(filePath) if files.endswith('.pickle')])
# 	foldSize = len(batches)
# 	for i in np.arange(foldSize): 
# 		print ('Running i is :', i)
		
# 		trainData = []
# 		trainLabels = []

# 		validData, validLabels = readFiles(filePath+batches[i])
		

# 		start = 0
# 		for no,j in enumerate(np.arange(foldSize)):
# 			if j!=i:
# 				# print ('Running j is : ', j)
# 				trn_d, trn_l = readFiles(filePath+batches[j]) 
# 				if start >0:
# 					trainData.append(trn_d) 
# 					trainLabels.append(trn_l)

# 		# print ('Validation Data and Labels shape: ', validData.shape, validLabels.shape)
# 		# print ('Training Data and Labels shape: ', trainData.shape, trainLabels.shape)
# 		yield np.array(trainData), np.array(trainLabels), np.array(validData), np.array(validLabels)


def genTrainValidFolds(filePath, oneHot=False):
	# stdBatchPath  = GlobalVariables()
	batches = np.array([files for files in os.listdir(filePath) if files.endswith('.pickle')])
	foldSize = len(batches)
	for i in np.arange(foldSize): 
		print ('Running i is :', i)
		
		validData, validLabels, labelDict = readFiles(filePath+batches[i])
		numLabels = len(np.unique(validLabels))
		if oneHot:
			validData, validLabels = reshape_data(dataset=validData, 
											labels=validLabels, 
											num_features=validData.shape[1], 
											numLabels=numLabels
											)

		start = 0
		for no,j in enumerate(np.arange(foldSize)):
			if j!=i:
				trn_d, trn_l, _ = readFiles(filePath+batches[j]) 

				if start == 0:   # Just dynamically initialize the arrays
					trainData = np.ndarray(shape=(trn_d.shape[0]*(foldSize-1),trn_d.shape[1]), dtype=np.float32)
					if oneHot:
						trainLabels = np.ndarray(shape=(trn_d.shape[0]*(foldSize-1),numLabels), dtype=np.float32)
					else:
						trainLabels = np.ndarray(shape=(trn_l.shape[0]*(foldSize-1),))

				if oneHot:
					trainData[start:start+trn_d.shape[0]], trainLabels[start:start+trn_d.shape[0]] = \
							reshape_data(dataset=trn_d, 
									labels=trn_l, 
									num_features=trainData.shape[1], 
									numLabels=numLabels
									)
				else:
					trainData[start:start+trn_d.shape[0]] = trn_d
					trainLabels[start:start+trn_l.shape[0]] = trn_l
							
				start = start+trn_d.shape[0]

		# print ('Validation Data and Labels shape: ', validData.shape, validLabels.shape)
		# print ('Training Data and Labels shape: ', trainData.shape, trainLabels.shape)
		yield trainData, trainLabels, validData, validLabels, labelDict



# STDbatch_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/STD/batchData/"
# EDGbatch_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/EDG/batchData/"
# HOGp1batch_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/HOGp1/batchData/"


# for trainData, trainLabels, validData, validLabels in genTrainValidFolds(HOGp1batch_dir, oneHot=False):
#     print ('Validation Data and Labels shape: ', validData.shape, validLabels.shape)
#     print ('Training Data and Labels shape: ', trainData.shape, trainLabels.shape)