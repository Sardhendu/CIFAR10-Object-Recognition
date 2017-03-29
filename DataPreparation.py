"""
Author: Sardhendu Mishra

Code Info: This module will go through all the folders representing each labels of the CIFAR10 dataset, slice and dice the data and create batches (10 batches), where batches are subset of data.

	1. All the batches have equally distributed  labels:
		For example batch 1 of size 5000 contains 500 dataset pertaining to label CAR, 500 dataset pertaining to cat and so on so forth.

"""


from __future__ import print_function
import os, sys, glob
import numpy as np

import cv2
import pickle
import random

# from FeatureExtraction import feature_pixelSTD
# # str()
# # 1234
    
# def create_dataset(dataPath, featureType, max_num_images=None,force_dump=None):
# 	for no,images_path in enumerate(dataPath):
# 		if max_num_images:
# 			file_names_arr = os.listdir(images_path)[0:max_num_images]
# 		else:
# 			file_names_arr = os.listdir(images_path)

# 		dataset = feature_pixelSTD(images_path,file_names_arr,image_size=32)

# 		# Store the image as a pickle file in the directory
# 		pickle_filename = os.path.basename(os.path.abspath(images_path))
# 		path_to_folder = conf['parent_dir']+'/'+featureType

# 		if not os.path.exists(path_to_folder):
# 			os.makedirs(path_to_folder)

# 		pickle_image_dir = os.path.dirname(os.path.abspath(images_path))+'/'+featureType+'/'+pickle_filename+'.pickle'
# 		# print (pickle_image_dir)
# 		if os.path.exists(pickle_image_dir) and not force_dump:
# 			print ('The path already exists, you should force the dump')
# 		else:
# 			try:
# 				with open(pickle_image_dir, 'wb') as f:
# 					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
# 			except Exception as e:
# 				print('Unable to save data to', pickle_image_dir, ':', e)



class CreateBatches():

	def __init__(self, dimensions):
		self.dimensions = dimensions
		# label_dict : keys are the object name and values are the labels assigned

	def initialize_arrays(self, num_images):
		dataset = np.ndarray((num_images, self.dimensions), dtype=np.float32)
		labels = np.ndarray(num_images, dtype=np.int32)
		return dataset, labels

	# You should provide the max_num_image. And the num_of_images per class label should not exceed it.
	# The below code created a big Training set with all the 
	def gen_TrainTestData(self, max_num_images, dir_to_pickle_files, test_percntg=10):
		
		label_categories = [files for files in os.listdir(dir_to_pickle_files) if files.endswith(".pickle")]
		# print (label_categories)

		if not os.path.exists(dir_to_pickle_files):
			print ('The path %s doesnt exists ', dir_to_pickle_files)
			# break

		test_size_per_class = int(test_percntg/100 * max_num_images)
		train_size_per_class = int(max_num_images - test_size_per_class)
		# print (train_size_per_class, test_size_per_class)

		# Get the num of train datapoint and num of test data points
		numTestPoints = int(np.ceil(test_size_per_class * len(label_categories)))
		numTrainPoints = int(train_size_per_class * len(label_categories))

		# print (numTestPoints)
		# print (numTrainPoints)
		
		testDataset, testLabels = self.initialize_arrays(numTestPoints)
		trainDataset, trainLabels = self.initialize_arrays(numTrainPoints)
		# print (testDataset.shape, testLabels.shape)
		# print (trainDataset.shape, trainLabels.shape)

		start_trn, start_tst = 0, 0
		end_trn, end_tst  = train_size_per_class, test_size_per_class
		end = train_size_per_class + test_size_per_class

		labelDict = {}
		# seed = 448
		seed = 8653
		random.seed(seed)
		print ('seed use for randomness is : ', seed )
		for label_id, label_file in enumerate(label_categories):
			# print (start_trn ,start_tst, end_trn, end_tst)
			labelDict[label_id] = label_file
			try:
				with open (os.path.join(dir_to_pickle_files,label_file), 'rb') as f:
					dataMatrix = pickle.load(f)
					# print (dataMatrix.shape)

					np.random.shuffle(dataMatrix)
					testDataset[start_tst:end_tst,:] = dataMatrix[0:test_size_per_class,:]
					trainDataset[start_trn:end_trn,:] = dataMatrix[test_size_per_class:end,:]
					testLabels[start_tst:end_tst] = label_id
					trainLabels[start_trn:end_trn] = label_id

					start_tst += test_size_per_class
					end_tst += test_size_per_class
					start_trn += train_size_per_class
					end_trn += train_size_per_class
			except Exception as e:
				print('Unable to process data from', pickle_files, ':', e)
				raise

		print ('The training Data set size is : ', trainDataset.shape)
		print ('The training Labels size is : ', trainLabels.shape)
		print ('The test Data set size is : ', testDataset.shape)
		print ('The test Labels size is : ', testLabels.shape)
		
		return 	trainDataset, trainLabels, testDataset, testLabels, labelDict


	def generateBatches(self, dataset, labels, numBatches=10):
		# The below three lines of code will just create a dummy array and jumble the indices so we get equal mumber of labels in all the batches.
		dummy_arr = np.arange(len(dataset))
		nwshape = (numBatches, int(len(dummy_arr)/numBatches))
		batch_indices = np.reshape(np.reshape(dummy_arr, nwshape).T, nwshape)

		# The below code will just provide the initiator function with the training dataset and labels for the input index values.
		for indices in batch_indices:
			yield dataset[indices], labels[indices]


	def dumpBatches(self, whereToDump, trnBatchData, trnBatchLabel, batchNum, labelDict=None,):
		print ('Batch No: ', batchNum, ' : Training Batch Data Shape:', trnBatchData.shape)
		print ('Batch No: ', batchNum, ' : Training Batch Labels Shape :', trnBatchLabel.shape)
				
		if not os.path.exists(whereToDump):
			os.makedirs(whereToDump)

		with open(whereToDump+'/'+str(batchNum)+'.pickle', 'wb') as f:
			batch = {
					'batchData': trnBatchData,
					'batchLabels': trnBatchLabel,
					'labelDict': labelDict
					}
			pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)
		


# imageDim = 32*32*3
# root_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/STD/"
# batch_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/STD/batchData/"
# obj_STD = CreateBatches(dimensions=imageDim)
# trainData, trainLabels, testLabels, _, _ = obj_STD.gen_TrainTestData(max_num_images=5000, dir_to_pickle_files=root_dir, test_percntg=0)
# for batchNum, (trnBatchData, trnBatchLabel) in enumerate(obj_STD.generateBatches(dataset=trainData, labels=trainLabels, numBatches=10)):
# 	obj_STD.dumpBatches(batch_dir, batchNum=batchNum)


# imageDim=32*32
# root_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/EDG/"
# batch_dir = "/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/Classification-1/EDG/batchData/"
# obj_EDG = CreateBatches(dimensions=imageDim)
# trainData, trainLabels, testLabels, _, _ = obj_EDG.gen_TrainTestData(max_num_images=5000, dir_to_pickle_files=root_dir, test_percntg=0)
# for batchNum, (trnBatchData, trnBatchLabel) in enumerate(obj_EDG.generateBatches(dataset=trainData, labels=trainLabels, numBatches=10)):
# 	obj_EDG.dumpBatches(batch_dir, batchNum=batchNum)
