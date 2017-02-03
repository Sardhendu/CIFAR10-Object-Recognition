
import os, sys,glob
import numpy as np
import pickle

# from Tools import GlobalVariables

stdBatchPath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/featureSTD/batchPath/'

def readFiles(dirIN):
    with open(dirIN, 'rb') as f:
        data = pickle.load(f)
    return data['batchData'], data['batchLabels']      

def reshape_data(dataset, labels, num_features, num_labels, sample_size=None):
    if sample_size:
        dataset = dataset[:sample_size].reshape(sample_size, num_features) # To reshape the  
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    else:
        dataset = dataset.reshape(len(dataset), num_features) # To reshape the  
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def genBatchIterator(self,batch_dir, batch_filename_arr):
    for no, batch_file in enumerate(batch_filename_arr):
        trnBatchData, trnBatchLabels = readFiles(os.path.join(batch_dir,batch_file))
        yield trnBatchData, trnBatchLabels

def genTrainValidFolds():
	# stdBatchPath  = GlobalVariables()
	batches = np.array([files for files in os.listdir(stdBatchPath) if files.endswith('.pickle')])
	foldSize = len(batches)
	for i in np.arange(foldSize): 
		print ('Running i is :', i)
		num_features = 32*32
		num_labels = 10
		trainData = np.ndarray(shape=(5000*(foldSize-1),num_features), dtype=np.float32)
		trainLabels = np.ndarray(shape=(5000*(foldSize-1),num_labels), dtype=np.float32)

		validData, validLabels = readFiles(stdBatchPath+batches[i])
		validData, validLabels = reshape_data(dataset=validData, 
                                               labels=validLabels, 
                                               num_features=num_features, 
                                               num_labels = num_labels)
		start = 0
		for no,j in enumerate(np.arange(foldSize)):
			if j!=i:
				# print ('Running j is : ', j)
				trn_d, trn_l = readFiles(stdBatchPath+batches[j])  
				trainData[start:start+trn_d.shape[0]], trainLabels[start:start+trn_l.shape[0]] = \
						reshape_data(dataset=trn_d, 
							labels=trn_l, 
							num_features=num_features, 
							num_labels = num_labels)
				start = start+trn_d.shape[0]

		# print ('Validation Data and Labels shape: ', validData.shape, validLabels.shape)
		# print ('Training Data and Labels shape: ', trainData.shape, trainLabels.shape)
		yield trainData, trainLabels, validData, validLabels